import cv2
from collections import deque
import gouda
import gouda.image as gimage
import gzip
import numpy as np
import os
import torch


def reload_model():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch',
                           'unet',
                           in_channels=3,
                           out_channels=1,
                           init_features=32,
                           pretrained=True,
                           verbose=False)
    model = model.eval().float()
    return model


def glorot_uniform(kernel_shape):
    """NumPy implementation for a glorot uniform kernel initializer"""
    receptive_field_size = np.prod(kernel_shape[:-2])
    fan_in = kernel_shape[-2] * receptive_field_size
    fan_out = kernel_shape[-1] * receptive_field_size

    np.random.seed(42)
    scale = 1. / max(1., (fan_in + fan_out) / 2)
    limit = np.sqrt(3.0 * scale)
    return np.random.uniform(-limit, limit, kernel_shape)


def save_arr(path, arr):
    if path.endswith('.gz'):
        with gzip.open(path, 'wb') as f:
            np.save(f, arr)
    else:
        np.save(path, arr)


def read_arr(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            data = np.load(f)
    elif path.endswith('.npy'):
        data = np.load(path)
    else:
        data = cv2.imread(path, -1)
    return data


def getattr_recursive(item, attr_string):
    """getattr that can be applied recursively

    Parameters
    ----------
    item : any
        The base item to examine
    attr_string : str
        A string with period-separated items - for example item.subitem.attribute

    """
    nested_type = type(item).__name__
    cur_item = item
    for key in attr_string.split('.'):
        try:
            cur_item = getattr(cur_item, key)
            nested_type += '.' + type(cur_item).__name__
        except AttributeError:
            raise AttributeError("'{}' object has no attribute '{}'".format(nested_type, key))
    return cur_item


def check_attr(obj, attr_string, default=None):
    if not hasattr(obj, attr_string):
        return default
    return getattr(obj, attr_string)


class LayerHooks:
    def __init__(self, model, layer, use_gradients=True):
        self.model = model
        self.layer = layer
        self.grads = deque()
        self.activations = []
        self.active = False
        self.use_gradients = use_gradients
        self.activation_hook = None
        self.gradient_hook = None
        self.start()

    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach().data)

    def save_gradient(self, module, input, output):
        if not check_attr(output, "requires_grad", False):
            raise ValueError("Cannot save gradient if the output doesn't have 'requires_grad'")

        def grad_hook(grad):
            self.grads.appendleft(grad.cpu().detach().data)
        output.register_hook(grad_hook)

    def start(self):
        self.activation_hook = self.layer.register_forward_hook(self.save_activation)
        if self.use_gradients:
            self.gradient_hook = self.layer.register_forward_hook(self.save_gradient)
        self.active = True

    def clear(self):
        self.activations = []
        self.grads = deque()

    def stop(self):
        if self.activation_hook is not None:
            self.activation_hook.remove()
            self.activation_hook = None
        if self.gradient_hook is not None:
            self.gradient_hook.remove()
            self.gradient_hook = None
        self.active = False

    def reset(self, restart=True):
        self.clear()
        self.stop()
        if restart:
            self.start()

    def get(self, as_numpy=True):
        activations = [self.activations[i].numpy() for i in range(len(self.activations))]
        grads = [self.grads[i].numpy() for i in range(len(self.grads))]
        return [activations, grads]


def compare_volumes(image_dir, return_volume=False):
    """Compare slices within a given image volume.

    There are 6 images missing the post-contrast channel and 10 images missing both pre- and post-contrast channels. In these cases, the FLAIR channel was duplicated onto the missing channel(s).
    """
    image_paths = gouda.get_sorted_filenames(os.path.join(str(image_dir), '*[1-9].tif'))

    images = []
    for path in image_paths:
        image = gimage.imread(path).transpose([2, 0, 1])
        images.append(image)
    images = np.stack(images, axis=0)
    if return_volume:
        return images

    check1 = np.abs(images[:, 0] - images[:, 1])
    check2 = np.abs(images[:, 1] - images[:, 2])
    check3 = np.abs(images[:, 0] - images[:, 2])
    return check1.sum(), check2.sum(), check3.sum()


def merge_maps(maps, output_shape=None, rescale=True):
    """Resize, rescale, and average the attribution maps together"""
    maps = [np.squeeze(item) for item in maps]
    if output_shape is None:
        output_shape = [0, 0]
        for item in maps:
            if item.size > np.prod(output_shape):
                output_shape = item.shape
    if rescale:
        maps = [gouda.rescale(cv2.resize(item, output_shape)) for item in maps]
    else:
        maps = [cv2.resize(item, output_shape) for item in maps]
    output_map = np.mean(maps, axis=0)
    return output_map


def prep_image(image):
    if not isinstance(image, np.ndarray):
        if image.endswith('.npy') or image.endswith('.npy.gz'):
            image = np.squeeze(read_arr(image))
        else:
            image = gimage.imread(image, -1)
            image = gouda.normalize(image, axis=(0, 1))
    image = image.transpose([2, 0, 1])[np.newaxis]
    image = torch.from_numpy(image.astype(np.float32))
    return image


def relu_arr(x):
    return x * (x > 0)
