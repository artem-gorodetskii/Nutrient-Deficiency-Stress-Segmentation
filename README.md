# Nutrient-Deficiency-Stress-Segmentation

This repository contains an implementation of a Unet-based model for segmentation of Nutrient Deficiency Stress (see [paper](https://arxiv.org/abs/2012.09654)).

### Dataset
The dataset is available at the [link](https://registry.opendata.aws/intelinair_longitudinal_nutrient_deficiency/):

### Model
The model is based on custom UNet architecture (see [model.py](https://github.com/artem-gorodetskii/Nutrient-Deficiency-Stress-Segmentation/blob/master/model.py)). Input to the model represents a combination of three RGB images.

### Evaluation
The dataset containing 386 flights was randonly separated on 328 train and 58 validation flights. The model was trained for only 26 epochs (see [train.py](https://github.com/artem-gorodetskii/Nutrient-Deficiency-Stress-Segmentation/blob/master/train.py) and [config.py](https://github.com/artem-gorodetskii/Nutrient-Deficiency-Stress-Segmentation/blob/master/config.py) for detail). 

The evalutaion results on validation data:
<div align="center">
  <img src="assets/evaluation.png" width="100" />
</div>
