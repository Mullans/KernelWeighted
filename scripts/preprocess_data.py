import argparse
import gouda
import gouda.image as gimage
import numpy as np
import torch
import tqdm.auto as tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import reload_model, save_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Kaggle Brain MRI data')
    parser.add_argument('--data_dir', default='data/kaggle_data', help='raw data directory')
    parser.add_argument('--preprocessed_dir', default='data/preprocessed', help='preprocessed data directory')
    parser.add_argument('--download', default=True, action=argparse.BooleanOptionalAction, help='download data from Kaggle')
    parser.add_argument('--foreground_size', default=100, help='minimum foreground size for image slices')
    parser.add_argument('--compress', default=True, action=argparse.BooleanOptionalAction, help='compress preprocessed data')
    args = parser.parse_args()

    raw_data = gouda.GoudaPath(args.data_dir)
    preprocessed = gouda.GoudaPath(args.preprocessed_dir).ensure_dir()
    foreground_size = args.foreground_size
    compress = args.compress
    # ---------------------------
    if args.download:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('mateuszbuda/lgg-mri-segmentation', path=args.data_dir, quiet=False, unzip=True)
    elif not raw_data.exists():
        raise ValueError('No data found at "{}"'.format(str(raw_data)))
    if raw_data('kaggle_3m').exists():
        raw_data = raw_data / 'kaggle_3m'

    model = reload_model().eval().cuda()
    with torch.no_grad():
        for volume_dir in tqdm.tqdm(raw_data.children()):
            mask_paths = volume_dir.glob('*_mask.tif')
            mask_paths = sorted(mask_paths, key=lambda x: int(gouda.basicname(x).rsplit('_', 2)[1]))
            image_paths = [path.replace('_mask', '') for path in mask_paths]

            image_slices = []
            for image_path in image_paths:
                image = gimage.imread(image_path, -1)
                # This shouldn't be used in the current dataset, but the model requires (256, 256) input
                if image.shape[:2] != (256, 256):
                    image = gimage.padded_resize(image, size=[256, 256], allow_rotate=False)
                image_slices.append(image)
            image_slices = np.stack(image_slices, axis=0).astype(np.float32)
            image_slices = gouda.normalize(image_slices, axis=(0, 1, 2))
            for i in range(len(image_slices)):
                image_slice = torch.from_numpy(image_slices[i:i + 1].transpose([0, 3, 1, 2]))
                pred_area = torch.sum(model(image_slice.cuda()))
                if pred_area < foreground_size:
                    continue
                dest_path = preprocessed.add_basename(image_paths[i]).replace('.tif', '.npy')
                if compress:
                    dest_path += '.gz'
                save_arr(dest_path, image_slices[i:i + 1].astype(np.float32))
