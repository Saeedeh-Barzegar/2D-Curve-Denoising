# Smooth by Design: Curve Denoising with Autoencoders

You can read the paper from [here](paper/Barzegar_CGVCVIP_2026.pdf).


**Abstract**:Curve data collected in practice is often contaminated with noise, which limits its reliability for analysis and visualization in downstream applications. Denoising such curves is therefore a crucial step for ensuring accurate interpretation and broader usability across domains. Autoencoders have proven effective in extracting meaningful representations from noisy data. In this work, we introduce an inception-based autoencoder that transforms an ordered sequence of noisy curve data into a smooth polygonal curve that closely preserves the structure of the original data. Unlike many existing approaches, our method does not impose restrictive assumptions on the global shape of the curve. Instead, it assumes that the data is sampled from an underlying smooth curve or surface. By leveraging this assumption, the model is able to reconstruct not only the geometry but also the differential properties of the original curve. The proposed architecture accommodates varying input sizes and learns robust representations of noisy inputs. Extensive experiments show that our inception-based design achieves superior performance compared to several classical and neural network-based denoising methods.

## Requirements

```
tensorflow >= 2.8
numpy
scipy
matplotlib
```

## How to Train

```shell
python train.py --noisy_dataset path/to/noisy --clean_dataset path/to/clean
```
