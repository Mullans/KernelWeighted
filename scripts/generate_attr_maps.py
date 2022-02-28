import argparse
import gouda
import gouda.image as gimage
import numpy as np
import os
import pandas as pd
import skimage.metrics
import torch
import tqdm.auto as tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.constants import BOTTLENECK, LAST_CONV, POST_BOTTLENECK_LAYERS
from src.utils import merge_maps, prep_image, reload_model, save_arr
from src.interpret import GradCAM, GradCAMPlus, ScoreCAM, KernelWeighted


def ordered_image_paths(data_dir):
    data_dir = gouda.GoudaPath(data_dir)
    all_images = []
    image_keys = [gouda.basicname(item) for item in data_dir.children()]
    for image_key in image_keys:
        all_images.extend(gouda.get_sorted_filenames(data_dir / image_key / image_key + '*[1-9].tif'))
    return all_images


def check_attr_method(attr_method, layers, preprocessed_dir, raw_data_dir, model=None, output_dir='', save_map=False, save_map_image=True, **kwargs):
    defaults = {
        'max_batch_size': 32,
        'min_pred_area': 100,
        'image_shape': (256, 256)
    }
    for key in defaults:
        if key not in kwargs:
            kwargs[key] = defaults[key]
    image_paths = gouda.GoudaPath(preprocessed_dir).glob('*.npy.gz')
    if model is None:
        model = reload_model()
    model = model.cuda()
    attr_model = attr_method(model, layers, max_batch_size=kwargs['max_batch_size'], verbose=False, use_cuda=True)
    attr_model.stop()
    output_dir = gouda.GoudaPath(output_dir).ensure_dir()
    log_path = output_dir('logfile.csv')
    total_results = []
    metrics = ['Path', 'LabelArea', 'PredArea', 'Dice', 'Hausdorff', 'MaskedPredArea', 'PredictionPreserved', 'ImagePreserved']
    log_template = '{:s},{:.2f},{:.2f},{:.4f},{:.4f},{:.2f},{:.4f},{:.4f}\n'

    with open(log_path, 'w') as log:
        log.write(','.join(metrics) + '\n')
        # pbar = tqdm.tqdm(total=len(image_paths))
        for path in tqdm.tqdm(image_paths):
            if '.npy.gz' in path:
                map_dest = output_dir.add_basename(path)
            else:
                map_dest = output_dir.add_basename(gouda.basicname(path) + '.npy.gz')
            image = prep_image(path)
            with torch.no_grad():
                pred = model(image.cuda()).cpu().detach().numpy()
            if pred.sum() < kwargs['min_pred_area']:
                continue
            attr_map = attr_model.forward(image, merge_layers=False)
            attr_model.stop()
            attr_map = merge_maps(attr_map, output_shape=kwargs['image_shape'])
            if save_map:
                save_arr(map_dest, attr_map.astype(np.float32))
            if save_map_image:
                gimage.imwrite(map_dest.replace('.npy.gz', '.png'), gouda.rescale(attr_map, 0, 255).astype(np.uint8))

            with torch.no_grad():
                attr_map = gouda.rescale(attr_map)
                masked_input = image * attr_map
                masked_pred = model(masked_input.cuda()).cpu().detach().numpy()

            basename = os.path.basename(path).split('.')[0]
            mask_path = gouda.GoudaPath(raw_data_dir) / basename.rsplit('_', 1)[0] / basename + '_mask.tif'
            mask = gouda.rescale(gimage.imread(mask_path, -1))
            pred_pres = np.minimum(pred, masked_pred).sum() / np.sum(pred)
            image_pres = np.mean(attr_map)
            pred = np.squeeze(pred)
            masked_pred = np.squeeze(masked_pred)

            results = {
                'Path': gouda.basicname(path),
                'LabelArea': mask.sum(),
                'PredArea': pred.sum(),
                'Dice': gouda.dice_coef(pred > 0.5, mask > 0.5),
                'Hausdorff': skimage.metrics.hausdorff_distance(pred > 0.5, mask > 0.5),
                'MaskedPredArea': masked_pred.sum(),
                'PredictionPreserved': pred_pres,
                'ImagePreserved': image_pres
            }
            total_results.append(results)
            log_line = log_template.format(*[results[key] for key in metrics])
            log.write(log_line)
    total_results = pd.DataFrame(total_results).set_index('Path')
    total_results.to_excel(str(output_dir / '{}_Results.xlsx'.format(output_dir.basicname())))
    return total_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Kaggle Brain MRI data')
    parser.add_argument('--data_dir', default='data/kaggle_data/kaggle_3m', help='raw data directory')
    parser.add_argument('--preprocessed_dir', default='data/preprocessed', help='preprocessed data directory')
    parser.add_argument('--output_dir', default='data/results', help='output results directory')
    args = parser.parse_args()

    raw_data = gouda.GoudaPath(args.data_dir)

    output_dir = gouda.GoudaPath(args.output_dir).ensure_dir()
    check_attr_method(GradCAM, BOTTLENECK, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'GradCAM_Bottleneck')
    check_attr_method(GradCAMPlus, BOTTLENECK, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'GradCAMPlus_Bottleneck')
    check_attr_method(ScoreCAM, BOTTLENECK, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'ScoreCAM_Bottleneck')

    check_attr_method(GradCAM, LAST_CONV, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'GradCAM_LastConv')
    check_attr_method(GradCAMPlus, LAST_CONV, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'GradCAMPlus_LastConv')
    check_attr_method(ScoreCAM, LAST_CONV, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'ScoreCAM_LastConv')

    check_attr_method(GradCAM, POST_BOTTLENECK_LAYERS, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'GradCAM_Adapted')
    check_attr_method(GradCAMPlus, POST_BOTTLENECK_LAYERS, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'GradCAMPlus_Adapted')
    check_attr_method(ScoreCAM, POST_BOTTLENECK_LAYERS, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'ScoreCAM_Adapted')

    check_attr_method(KernelWeighted, POST_BOTTLENECK_LAYERS, args.preprocessed_dir, args.data_dir, output_dir=output_dir / 'KernelWeighted')

    # Merge the results into a single sheet
    merged_df = None
    for path in output_dir.glob('**/*_Results.xlsx', sort=True, as_gouda=True):
        results_df = pd.read_excel(path, index_col=0)
        column_map = {'PredictionPreserved': '{}_PP'.format(path[-2]), 'ImagePreserved': '{}_IP'.format(path.basicname())}
        if merged_df is None:
            merged_df = results_df.rename(columns=column_map)
            continue
        else:
            merged_df = merged_df.join(results_df[['PredictionPreserved', 'ImagePreserved']]).rename(columns=column_map)
    merged_df.to_excel(output_dir / 'MergedAttrResults.xlsx')
