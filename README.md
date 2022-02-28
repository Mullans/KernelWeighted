# Visual attribution for deep learning segmentation in medical imaging
This code was used in the work presented at [SPIE Medical Imaging 2022](https://spie.org/medical-imaging/presentation/Visual-attribution-for-deep-learning-segmentation-in-medical-imaging/12032-25)


## Data/Model Links
The dataset was made available by Buda et al. at:
https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

The PyTorch implementation of the pre-trained segmentation model was made available by Buda et al. at:
https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet

### Data Preparation
Run `python scripts/preprocess_data.py` to generate each image slice normalized by volume. You can use `python scripts/preprocess_data.py -h` to see preprocessing options that are available.


## Requirements
* gouda (0.5.5+)
* __numpy__ (1.21.2)*
* opencv (4.5.3)
* pandas (1.2.4+)
* __pytorch__ (1.8+)*
* scikit-image (0.18.1+)
* tqdm (4.59.0)

*The basic attribution methods can be used with only __numpy__ and __pytorch__. The other requirements are used for methods such as image resizing, io methods, and other utilities.



## Usage
There are 4 available segmentation attribution methods available: GradCAM, GradCAM++, ScoreCAM, and KernelWeighted Contribution. Each method is initialized with a pytorch model and a list of layers to analyze. The layers can be strings if the model uses named tensors (ex: `["decoder2.dec2conv1"]`)

        attribution_model = interpret.KernelWeighted(model, layers)

The `.forward` method of the wrapped model can be used to get the attribution map for a given image, or the `.__call__` method can be used to get the predicted segmentation.

        attribution_map = attribution_model.forward(image)
        segmentation_map = attribution_model(image)
