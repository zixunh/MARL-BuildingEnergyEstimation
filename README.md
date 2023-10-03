# MARL: [M]ulti-scale [A]rchetype [R]epresentation [L]earning and Clustering for Building Energy Estimation
This repository is the implementation code of the paper "MARL: [M]ulti-scale [A]rchetype [R]epresentation [L]earning and Clustering for Building Energy Estimation" ([arxiv](https://arxiv.org/abs/2310.00180), poster).

![Frame_2](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/assets/106426767/f03687f2-044c-48f5-818e-27b1f70a92cb)

We present Multi-scale Archetype Representation Learning (MARL), a method designed to **automate local building archetype** construction through representation learning. Our proposed method addresses the aforementioned challenges by refining the essential elements of building archetypes for **Urban Building Energy Modeling**. This is a **learning-based** pipeline for representing and clustering buildings in our urban environment. Our research can be used in building energy estimation and can significantly save computing time. 

### Recent Updates:
- 08/08/23: Our work is accepted to **ICCV 2023 Workshop**: 1st Computer Vision Aided Architectural Design (CVAAD) Workshop.

### Dataset:
- For footprints and their meta info, please refer to this [folder](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/tree/main/data).
- For building energy consumption data, all rights are reserved by the [Lawrence Berkeley National Lab](https://buildings.lbl.gov/). Please contact authors for more detailed information.

### Requirements:
Before running our data generation and annotation pipeline, you can activate a conda environment where Python Version >= 3.7:
```
conda create --name [YOUR ENVIR NAME] python = [PYTHON VERSION]
conda activate [YOUR ENVIR NAME]
```
then install all necessary packages:
```
pip install -r requirements.txt
```

### Train:
To run training of our model, please refer to [this notebook](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/blob/main/notebooks/train_marl.ipynb), or run the following command:
```
python train.py
```
### Archetype Clustering:
To get latent representation and run clustering, please refer to [this notebook](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/blob/main/notebooks/latent_clustering.ipynb).
### Citation
If our work is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```
@misc{zhuang2023marl,
      title={MARL: Multi-scale Archetype Representation Learning for Urban Building Energy Modeling}, 
      author={Xinwei Zhuang and Zixun Huang and Wentao Zeng and Luisa Caldas},
      year={2023},
      eprint={2310.00180},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

