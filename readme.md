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


### 👉Reference
```
@article{van2017neural,
  title={Neural discrete representation learning},
  author={Van Den Oord, Aaron and Vinyals, Oriol and others},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

---
# 2.0 multi-scale vqae
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



