# MARL: [M]ulti-scale [A]rchetype [R]epresentation [L]earning and Clustering for Building Energy Estimation
This repository is the implementation code of the paper "MARL: [M]ulti-scale [A]rchetype [R]epresentation [L]earning and Clustering for Building Energy Estimation" (arxiv, poster, dataset).

![Frame_2](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/assets/106426767/f03687f2-044c-48f5-818e-27b1f70a92cb)

We present Multi-scale Archetype Representation Learning (MARL), a method designed to **automate local building archetype** construction through representation learning. Our proposed method addresses the aforementioned challenges by refining the essential elements of building archetypes for **Urban Building Energy Modeling**. This is a **learning-based** pipeline for representing and clustering buildings in our urban environment. Our research can be used in building energy estimation and can significantly save computing time. 

### Recent Updates:
- 08/08/23: Our work is accepted to ICCV 2023 Workshop: 1st Computer Vision Aided Architectural Design Workshop.

# Notes on LA data

current # data = 1,048,575
height info is color coded with viridis 
with Residential only, within the residential category, 
The categories are coded as below:

| Index | Category |
| ------------- | ------------- |
|0|Mobile Homes|
|1|Three Units (Any combination)|
|2|Two Units|
|3|Single|
|4|Five or more apartments|
|5|Four Units (Any combination)|
|6|Mobile Home Parks|
|7|Rooming Houses|


# Notes on SF data
current # data = 2806, with several empty data
<b>metadata</b>  

[!] need to remove <b>0</b> and <b>null</b> value in year, which corresponds to 999 in category
The meta data has three columns:

|index|year|category|
| ------------- | ------------- |---|
|image index for floorplan|construction year|see below|

The building categories are coded as below:
| Index | Category |
| ------------- | ------------- |
|0|CIE|
|1|Residential|
|2|MIPs|
|3|Hotel|
|4|MixRes|
|5|Mixed|
|6|MED|
|7|PDR|
|8|Retail/Entertain|
|999|Null|

### Citation
```

```

