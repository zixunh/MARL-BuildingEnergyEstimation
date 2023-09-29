# Dataset Config

1. You can download the processed LA floorplan from this [folder](https://drive.google.com/drive/folders/1wVxFvvGAp8CeAzttWhccXPBq4k0_mc-n?usp=sharing).
2. Unzip the compressed file and create a softlink to the current folder.
```
|_data
    |_data_root
        |_data01
        |_data02
        ...
```

# Dataset Info 
Our dataset includes footprints in the following 6 regions:
![regionmapfinal](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/assets/106426767/076c223a-64cc-481a-afdf-c7907f60ae06)

The corresponding meta information is stored [here](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/blob/main/data/data_config/meta.csv). The categories are coded as below:

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

