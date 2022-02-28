import numpy as np
import torch
import tqdm.auto as tqdm
from .utils import getattr_recursive, glorot_uniform, LayerHooks, merge_maps, relu_arr


class AttributionModel:
    def __init__(self, model, layers, use_cuda=True, max_batch_size=16, verbose=1, output_activation=None, use_gradients=True, **kwargs):
        self.model = model.eval()
        self.use_cuda = use_cuda
        self.layers = []
        self.acts_and_grads = []
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        if isinstance(layers, str) or not hasattr(layers, '__iter__'):
            layers = [layers]
        for layer in layers:
            if isinstance(layer, str):
                # This is used for named tensors such as 'decoder1.decoder2'
                layer = getattr_recursive(self.model, layer)
            self.layers.append(layer)
            self.acts_and_grads.append(LayerHooks(self.model, layer, use_gradients=use_gradients))
        self.use_gradients = use_gradients
        self.output_activation = output_activation

    def forward(self, image, **kwargs):
        if self.use_cuda:
            image = image.cuda()
            self.model = self.model.cuda()
        self.reset()
        output = self(image)
        if self.use_gradients:
            self.model.zero_grad()
            output.backward(torch.ones_like(output), retain_graph=False)
        att_maps = []
        self.stop()
        iterator = zip(self.layers, self.acts_and_grads)
        if self.verbose == 1:
            iterator = tqdm.tqdm(iterator, total=len(self.layers))
        for layer, acts_and_grads in iterator:
            acts, grads = acts_and_grads.get()
            if self.output_activation is not None:
                acts = [self.output_activation(act) for act in acts]
            weights = self.get_layer_weights(input_image=image, activations=acts, gradients=grads, predictions=output, layer=layer, **kwargs)
            # sum over 0-images, 1,2-channels
            att_map = np.sum(weights[:, np.newaxis] * acts, axis=(0, 1, 2))
            att_map = np.maximum(att_map, 0)
            att_maps.append(att_map)
        return att_maps

    def get_layer_weights(self, input_image=None, activations=None, gradients=None, predictions=None, **kwargs):
        raise NotImplementedError('get_layer_weights is only implemented in subclasses of InterpModel')

    def start(self):
        for item in self.acts_and_grads:
            item.start()

    def stop(self):
        for item in self.acts_and_grads:
            item.stop()

    def clear(self):
        for item in self.acts_and_grads:
            item.clear()

    def reset(self, restart=True):
        for item in self.acts_and_grads:
            item.reset(restart=restart)

    def __call__(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if self.output_activation:
            output = self.output_activation(output)
        return output


class GradCAM(AttributionModel):
    def get_layer_weights(self, gradients=None, **kwargs):
        weights = np.mean(gradients[-1], axis=(2, 3), keepdims=True)
        return weights

    @property
    def name(self):
        return 'GradCAM'


class GradCAMPlus(AttributionModel):
    """adapted from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py"""
    def get_layer_weights(self, activations=None, gradients=None, **kwargs):
        gradients = gradients[-1]
        activations = activations[-1]
        grads_2 = gradients ** 2
        grads_3 = grads_2 * gradients
        sum_acts = np.sum(activations, axis=(2, 3), keepdims=True)
        eps = 1e-6
        aij = grads_2 / (2 * grads_2 + sum_acts * grads_3 + eps)
        weights = np.maximum(gradients, 0) * aij
        weights = np.sum(weights, axis=(2, 3), keepdims=True)
        return weights

    @property
    def name(self):
        return 'GradCAM++'


class ScoreCAM(AttributionModel):
    def __init__(self, model, layers, use_gradients=False, verbose=1, **kwargs):
        super().__init__(model, layers, use_gradients=False, verbose=verbose, **kwargs)

    def get_layer_weights(self,
                          input_image=None,
                          activations=None,
                          predictions=None,
                          zero_baseline=True,
                          low_memory=False,
                          **kwargs):
        activations = activations[-1]
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=input_image.shape[-2:])
            acts = torch.from_numpy(activations)
            if self.use_cuda:
                self.model = self.model.cuda()
                if not low_memory:
                    acts = acts.cuda()
            if low_memory:
                input_image = input_image.cpu()

            max_val = acts.view(*acts.shape[:2], -1).max(dim=-1).values[:, :, None, None]
            min_val = acts.view(*acts.shape[:2], -1).min(dim=-1).values[:, :, None, None]
            acts = (acts - min_val) / (max_val - min_val)
            upsampled = upsample(acts)
            if not low_memory:
                masked_images = input_image[:, None] * upsampled[:, :, None]
            scores = []
            model_output = predictions.cpu().numpy()
            if self.verbose == 2:
                pbar = tqdm.tqdm(total=input_image.shape[0] * upsampled.shape[1])
            for image_idx in range(input_image.shape[0]):
                for idx in range(0, upsampled.shape[1], self.max_batch_size):
                    if low_memory and self.use_cuda:
                        masked_batch = input_image[image_idx, None].cuda() * upsampled[image_idx, idx:idx + self.max_batch_size, None].cuda()
                    else:
                        masked_batch = masked_images[image_idx, idx:idx + self.max_batch_size]
                    masked_prediction = self.__call__(masked_batch).cpu().numpy()
                    if zero_baseline:
                        batch_scores = np.sum(masked_prediction, axis=(1, 2, 3))
                    else:
                        batch_scores = np.sum(masked_prediction - model_output, axis=(1, 2, 3))
                    scores.extend(batch_scores)
                    if self.verbose == 2:
                        pbar.update(masked_batch.shape[0])
            if self.verbose == 2:
                pbar.close()
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[:2])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights[:, :, None, None]

    @property
    def name(self):
        return 'ScoreCAM'


class KernelWeighted(AttributionModel):
    def __init__(self, model, layers, merge_layers=True, use_gradients=False, output_activation=relu_arr, verbose=1, **kwargs):
        super().__init__(model, layers, use_gradients=False, output_activation=output_activation, verbose=verbose, **kwargs)
        self.merge_layers = merge_layers

    def forward(self, image, merge_layers=True, **kwargs):
        layer_maps = super().forward(image, **kwargs)
        if merge_layers:
            return merge_maps(layer_maps, rescale=False)
        else:
            return layer_maps

    def get_layer_weights(self, input_image=None, activations=None, predictions=None, layer=None, **other):
        baseline_weight = 1. / torch.sum(predictions, (1, 2, 3))
        if isinstance(layer, torch.nn.Conv2d):
            output_axis = 0
        elif isinstance(layer, torch.nn.ConvTranspose2d):
            output_axis = 1
        else:
            raise ValueError('KernelWeighted has only been implemented for conv2d and convtranspose2d layers right now')

        kernel = torch.clone(layer.weight)
        empty_kernel = torch.from_numpy(glorot_uniform(kernel.shape)).float()
        if self.use_cuda:
            empty_kernel = empty_kernel.cuda()

        weights = torch.zeros([input_image.shape[0], kernel.shape[output_axis]])
        idx_slice = [slice(None) for _ in range(kernel.ndim)]
        iterator = tqdm.trange(kernel.shape[output_axis]) if self.verbose == 2 else range(kernel.shape[output_axis])
        with torch.no_grad():
            for idx in iterator:
                temp_kernel = torch.clone(kernel).detach()
                idx_slice[output_axis] = idx
                temp_kernel[idx_slice] = empty_kernel[idx_slice]
                layer.weight = torch.nn.Parameter(temp_kernel, requires_grad=False)
                dependent_pred = self(input_image)
                dependent_contribution = torch.sum(predictions - torch.minimum(dependent_pred, predictions),
                                                   dim=(1, 2, 3), keepdim=True)

                temp_kernel = torch.clone(empty_kernel)
                temp_kernel[idx_slice] = kernel[idx_slice]
                layer.weight = torch.nn.Parameter(temp_kernel, requires_grad=False)
                independent_pred = self(input_image)
                independent_contribution = torch.sum(torch.minimum(independent_pred, predictions),
                                                     dim=(1, 2, 3), keepdim=True)

                weight = (dependent_contribution * independent_contribution) * baseline_weight
                weight = weight[:, 0, 0, 0]  # weight will always have shape: [batch, 1, 1, 1]
                weights[:, idx] = weight
            layer.weight = torch.nn.Parameter(kernel)
            weights = (weights - torch.min(weights)) / (torch.max(weights) - torch.min(weights))
        return weights.cpu().numpy()[:, :, None, None]

    @property
    def name(self):
        return 'KernelWeighted'
