<p align="center">
  <img width="50%" src="graphics/UCT_banner.jpeg" alt="UCT Banner"/>
</p>

## Authors and Rights
Student: Winner Kazaka - CS Honours
Supervisor: Associate Professor Geoff Nitschke
Organistion: University of Cape Town

## Abstract
In the field of Orthopedic Pathology, expert diagnoses can be prone to errors due to human limitations, and having specialized teams is not always feasible. Conversely, machine learning (ML) algorithms excel in image recognition tasks, akin to those performed by medical specialists. Nonetheless, the scarce availability of extensive datasets in the medical field can hamper the optimal performance of these algorithms. This research assesses the efficacy of state-of-the-art (SOTA) Supervised Learning models and strategies, namely Data Augmentation (DA) and Transfer Learning (TL), in enhancing classification accuracy on two medical datasets. The results establish that ConvNeXts are the most proficient image classifiers among the tested models and underscores the effectiveness of pre-training and fine-tuning methodologies in optimizing Deep Learning (DL) models. While DA shows marginal benefits, particularly through Neural Augment, its impact on model accuracy is not substantial. Importantly, the introduction of a new validation technique, Default Validation, notably improves model accuracy by a margin of $1.6\%$ to $6.9\%$.
    
## Installation and Setup

You can set up and run the project using the following commands:

```sh
# Installation
make install

# Virtual Environment Setup
make venv

# Validate on test set
make test
```

## Usage 

To run the script with specific configurations, you can use the command line arguments to specify the various parameters.
```sh
python3 src/main.py --model_names <model_name1,model_name2,...> --dir_paths <dir_path1,dir_path2,...> --data_augmentations <data_augmentation1,data_augmentation2,...> --transfer_learning_methods <method1,method2,...>
```
### Arguments:

--model_names: Specifies the models to use. Possible options include:

* convnext_tiny
* convnext_small
* convnext_base
* swin_v2_t
* swin_v2_s
* swin_v2_b
* densenet121
* densenet169
* densenet201
* resnet18
* resnet50
* resnet152
* efficientnet_v2_s
* efficientnet_v2_m
* efficientnet_v2_l

--dir_paths: Specifies the directory paths for the datasets. Example options:

* ./dataset/Neck
* ./dataset/lbow

--data_augmentations: Specifies the data augmentation methods to use. Possible options are:

* no_da
* RandomCrop
* RandAug
* NeuralAug

--transfer_learning_methods: Specifies whether to use transfer learning and the method to use. Possible options are:

* pretrained
* not_pretrained

### Example Usage:
```sh
python src/main.py --model_names convnext_base,densenet121 --dir_paths ./dataset/Neck,./dataset/lbow --data_augmentations RandomCrop,RandAug --transfer_learning_methods pretrained,not_pretrained
```

## Code Reference List

- <img src="graphics/github_logo.png" alt="Github Logo" width="20px"/> [Neural Augment](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/neuralstyle)

- <img src="graphics/github_logo.png" alt="Github Logo" width="20px"/> [Exact Match](https://gist.github.com/jadhavpritish/1991d808ac4cab908912455178848493#file-one_zero_loss-py)

- <img src="graphics/kaggle_logo.webp" alt="Kaggle Logo" width="20px"/> [Transfer Learning](https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch)

- <img src="graphics/kaggle_logo.webp" alt="Kaggle Logo" width="20px"/> [Multi-label Classification](https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset/code)


## Top Models and Baseline Weights

Access the top models and baseline weights here:

- [Trained Models](https://drive.google.com/drive/folders/10W2Gx_Yr3f87Cn1z-3hi-7lZKcVjNqkw?usp=sharing)

## Datasets

Access the Neck and Elbow datasets

- [Datasets](https://drive.google.com/file/d/1Cmvqm1GQw_rIxjMuqTeyeuRGL2ObVOl-/view?usp=drive_link)


