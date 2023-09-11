<p align="center">
  <img width="50%" src="graphics/UCT_banner.jpeg" alt="UCT Banner"/>
</p>

## Abstract
In the field of Orthopedic Pathology, expert diagnoses can be prone to errors due to human limitations, and having specialized teams is not always feasible. Conversely, machine learning (ML) algorithms excel in image recognition tasks, akin to those performed by medical specialists. Nonetheless, the scarce availability of extensive datasets in the medical field can hamper the optimal performance of these algorithms. This research assesses the efficacy of state-of-the-art (SOTA) Supervised Learning models and strategies, namely Data Augmentation (DA) and Transfer Learning (TL), in enhancing classification accuracy on two medical datasets. The results establish that ConvNeXts are the most proficient image classifiers among the tested models and underscores the effectiveness of pre-training and fine-tuning methodologies in optimizing Deep Learning (DL) models. While DA shows marginal benefits, particularly through Neural Augment, its impact on model accuracy is not substantial. Importantly, the introduction of a new validation technique, Default Validation, notably improves model accuracy by a margin of $1.6\%$ to $6.9\%$.
    
## Installation and Setup

You can set up and run the project using the following commands:

```sh
# Installation
make install

# Virtual Environment Setup
make venv

# Run the Training Pipeline
make run

# Run the Testing Pipeline
make test
```

## Code Reference List

- <img src="graphics/github_logo.png" alt="Github Logo" width="20px"/> [Neural Augment](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/neuralstyle)

- <img src="graphics/github_logo.png" alt="Github Logo" width="20px"/> [Exact Match](https://gist.github.com/jadhavpritish/1991d808ac4cab908912455178848493#file-one_zero_loss-py)

- <img src="graphics/kaggle_logo.webp" alt="Kaggle Logo" width="20px"/> [Transfer Learning](https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch)

- <img src="graphics/kaggle_logo.webp" alt="Kaggle Logo" width="20px"/> [Multi-label Classification](https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset/code)

## Datasets

Find the datasets in the following links:

- [Elbow Dataset](https://github.com/bryankazaka/DEEPPC-Supervised-Learning/src/trained_models/dataset/Elbow)
- [Neck Dataset](https://github.com/bryankazaka/DEEPPC-Supervised-Learning/src/trained_models/dataset/Neck)

## Top Models and Baseline Weights

Access the top models and baseline weights here:

- [Trained Models](https://github.com/bryankazaka/DEEPPC-Supervised-Learning/models/trained_models)

