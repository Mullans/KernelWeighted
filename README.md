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


## References
* Buda, M., Saha, A., and Mazurowski, M. A., “Association of genomic subtypes of lower-grade gliomas
with shape features automatically extracted by a deep learning algorithm,” Computers in Biology and
Medicine 109, 218–225 (2019).
* [GradCAM] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D., “Grad-cam: Visual
explanations from deep networks via gradient-based localization,” in [Proceedings of the IEEE international
conference on computer vision ], 618–626 (2017).
* [GradCAM++] Chattopadhay, A., Sarkar, A., Howlader, P., and Balasubramanian, V. N., “Grad-cam++: Generalized
gradient-based visual explanations for deep convolutional networks,” in [2018 IEEE Winter Conference on
Applications of Computer Vision (WACV) ], 839–847 (2018).
* [ScoreCAM] Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., and Hu, X., “Score-cam:
Score-weighted visual explanations for convolutional neural networks,” in [Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition workshops ], 24–25 (2020).
