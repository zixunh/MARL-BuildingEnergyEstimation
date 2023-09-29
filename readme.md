# MARL: [M]ulti-scale [A]rchetype [R]epresentation [L]earning and Clustering for Building Energy Estimation
We present Multi-scale Archetype Representation Learning (MARL), a method designed to **automate local building archetype** construction through representation learning. Our proposed method addresses the aforementioned challenges by refining the essential elements of building archetypes for **Urban Building Energy Modeling**. This is a **learning-based** pipeline for representing and clustering buildings in our urban environment. Our research can be used in building energy estimation and can significantly save computing time. 
![Frame_2](https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation/assets/106426767/f03687f2-044c-48f5-818e-27b1f70a92cb)

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


The categories are coded as below:

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

# Version 1.0
### TODO
- [x] vqae
- [ ] energy analyse
- [ ] conf paper
- [ ] code release and dependency

### 👉[Concept Warmup](https://github.com/ZixunHuang1997/VQVAE-Archetype/blob/main/review-generative-sol.pdf)
1. Reparameterize Trick
2. Latent Variable Models
3. AE, VAE, VQAE, VQVAE


### 👉[Paper Draft](https://docs.google.com/document/d/1ge4OY-r1BiU2jtaeFwXgkGBtqrn0BLNq0VhEswCeo_Y/edit?usp=sharing)

### 👉[Get Started with notebook](https://github.com/ZixunHuang1997/VQVAE-Archetype/blob/main/train.ipynb)

### 👉[Best Checkpoint Released](https://github.com/ZixunHuang1997/VQVAE-Archetype/tree/main/best_checkpoint)
- Related hyper-parameters is included in the notebook
- best_loss = 0.21704871313912527

### 👉Loss
##### train
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/4c672a49-da2f-4157-bbfb-9d74de229ebc)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/b8b03c4c-3066-410e-b958-60db297fe3d4)

### 👉Floor Plan Recon
##### case 1
![download](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/027d137d-6c23-46ec-809a-39c0f3b67e71)
![download](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/8ff2786b-83b7-4391-a528-66210b659e47)
##### case 2
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/95b3e876-120d-41a3-9d94-7a49f562512a)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/5d688ebe-b08d-44ea-b8df-33464c6a7fba)

## K-means Clustering (K=16, for all type building)
### latent visualization with tSNE
![download](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/4691db79-58d8-4d50-ac99-5b57d3e3224a)
### latent clustering with KMeans
![download](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/a070de53-23dc-4efa-aeb8-8c30ee306f9a)
### Decode the cluster centers
![download](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/9699aeea-d2ff-451f-a8e4-761fd527d381)

## Todo: residential

### Citation
```

```

---
# Version 2.0
### TODO
- [x] multiscaled vqae
- [ ] ae compare
- [ ] Jou paper draft
- [ ] add meta data

### multi-scale vqae loss
- best loss = 0.14986537524632043

![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/b916ff8a-be6f-4516-993f-1a1d86f2cecd)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/95f12321-cb38-4d5f-b07e-eba199fdb6e3)

### vqae multi-scaled recon for buildings with large floor area
##### case 1
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/1f805e21-cbf2-41bd-aada-6eaa3bb14c81)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/16e75f2e-e331-4453-9426-00bde4bd66e1)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/0f116e39-352f-4547-9b23-1def423d47fa)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/80bbd429-7ab4-4c11-a359-002e1357705b)
##### case 2
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/3fa96413-2144-4730-881c-0b3ff319cb2c)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/1f1ea28a-b48c-4c63-9910-9e82c2c5a6bf)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/87199321-1486-43cf-b790-b03d88b58d71)
![image](https://github.com/ZixunHuang1997/VQVAE-Archetype/assets/106426767/853063a0-9487-464f-8325-41f8f31b42c7)



