# Semantic feature Extraction on Perturbed Images

To evaluate the effectiveness of frequency domain-based methods in semantic extraction, we investigate the semantic changes caused by adding different perturbations to the samples. Two types of perturbed data were evaluated, including adversarial samples and Out-of-distribution (OOD) samples. 

To generate adversarial samples, we utilized the pre-trained VGG16 model on ImageNet as the under attack model, then we created adversarial samples using the advanced adversarial attack techniques PGD and CW. 

To generate OOD samples, we used widely recognized corrupted benchmarking, which includes 19 types of corruption, involving changes in weather, blur, noise and digital. 

This file present some examples of semantic feature extraction.

## 1. Adversarial attack (Evaluation on adversarial samples) 

### 1.1 PGD attack
Image with lot of textures.
![](./figs/pgd_1.png)
Image with cluttered backgrounds.
![](./figs/pgd_2.png)
Image with single object.
![](./figs/pgd_3.png)
Image with Bright Colors.
![](./figs/pgd_4.png)
Image with flat Colors.
![](./figs/pgd_5.png)

### 1.2 CW attack

![](./figs/cw_1.png)

![](./figs/cw_2.png)

![](./figs/cw_3.png)

![](./figs/cw_4.png)

![](./figs/cw_5.png)

## 2. Corruption (Evaluation on OOD samples) 

### 2.1 noise
gaussian_noise
![](./figs/gaussian_noise.png)
shot_noise
![](./figs/shot_noise.png)
impulse_noise
![](./figs/impulse_noise.png)

### 2.2 blur
defocus_blur
![](./figs/defocus_blur.png)
glass_blur
![](./figs/glass_blur.png)
motion_blur
![](./figs/motion_blur.png)
zoom_blur
![](./figs/zoom_blur.png)

### 2.3 weather
frost
![](./figs/frost.png)
snow
![](./figs/snow.png)
fog
![](./figs/fog.png)
brightness
![](./figs/brightness.png)

### 2.4 digital

contrast
![](./figs/contrast.png)
elastic_transform
![](./figs/elastic_transform.png)
pixelate
![](./figs/pixelate.png)
jpeg_compression
![](./figs/jpeg_compression.png)

### 2.5 extra

speckle_noise
![](./figs/speckle_noise.png)
spatter
![](./figs/spatter.png)
gaussian_blur
![](./figs/gaussian_blur.png)
saturate
![](./figs/saturate.png)