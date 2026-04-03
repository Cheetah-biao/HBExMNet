# HBExMNet

HBExMNet is the Python training and inference project for multi-stage volumetric image restoration.

The current release supports:

- denoise training
- super-resolution training
- chained denoise plus super-resolution inference
- GUI-based operation
- command-line operation

A matched standalone Windows C++ inference package is also released separately for users who prefer direct deployment without installing a Python environment.

Recommended hardware for the standalone Windows software:

- 24 GB GPU memory recommended
- 64 GB system RAM recommended
- a working NVIDIA driver
- tested on NVIDIA GeForce RTX 3090 and RTX 4090

## Project Layout

This repository is intended to be uploaded directly as a single project folder named `HBExMNet`.

The expected final layout is:

```text
HBExMNet/
|-- Inference.py
|-- Train.py
|-- README.md
|-- requirements.txt
|-- data
|-- docs
|   `-- images
|-- experiments
|-- models
|-- options
`-- utils
```

- pretrained models under `experiments/` stay inside `HBExMNet`

## Environment

The Python workflow has been tested with:

- Windows 10
- Python 3.9.12 (64-bit)
- PyTorch 1.11.0
- Intel Core i9-10900X CPU @ 3.70 GHz
- NVIDIA GeForce RTX 3090 / 4090

## Installation

1. Create a Python 3.9.12 environment:

```bash
conda create -n HBExMNet python=3.9.12
conda activate HBExMNet
```

2. Install PyTorch:

```bash
conda install pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Install the remaining dependencies from the project root:

```bash
pip install -r requirements.txt
```

4. Download the pretrained models and the test data package from the links below.

- The pretrained model package must be extracted into the `HBExMNet` project folder so that the model files are placed under `HBExMNet/experiments/`. The inference and training code will not run correctly unless the pretrained models are available there.
- The test data package can be extracted to any convenient local folder. During inference, simply select that folder in the GUI or pass it with `--input-path`.
- A training data package is not included in this release. Prepare your own training folders when needed.

The GUI uses the bundled models under `HBExMNet/experiments/`. Test data does not need to be placed inside the project folder.

## Model Zoo And Data Download

Large files are intentionally distributed outside the Git repository.

| Asset | Content | Extract To | Link |
| --- | --- | --- | --- |
| Pretrained models | trained checkpoints under `experiments/` | `HBExMNet/experiments/` | <div align="center"><a href="https://drive.google.com/file/d/1lyK_Xlty2XjyPswDPf9otiYXmex_DsiA/view?usp=sharing"><img src="https://img.shields.io/badge/Models-4285F4?style=flat-square&logo=googledrive&logoColor=white" alt="Models" /></a></div> |
| Test data package | example TIFF stacks for inference | sibling folder `data/` beside `HBExMNet/` | <div align="center"><a href="https://drive.google.com/file/d/1mYuX0no779IjG9qgwa8DzHmhW4rrSEyC/view?usp=drive_link"><img src="https://img.shields.io/badge/Data-4285F4?style=flat-square&logo=googledrive&logoColor=white" alt="Data" /></a></div> |
| Standalone Windows software | compiled C++ TensorRT inference package | any local folder | <div align="center"><a href="https://drive.google.com/file/d/1y4zZPHFt4tSbWfCmNTZYXznky_g2y7J3/view?usp=sharing"><img src="https://img.shields.io/badge/Software-4285F4?style=flat-square&logo=googledrive&logoColor=white" alt="Software" /></a></div> |


## Quick Start

### Training

Run the following command from the project root:

```powershell
python Train.py
```

Example training GUI:

<div align="center">
  <img width="760" src="docs/images/Train.png" />
</div>

In the training GUI, configure:

- `Mode`: `TIM` or `SDCM`
- `Organelle`
- `Task`: `Denoise` or `SR`
- training folders for `LR` and `GT`
- patch parameters

### Inference

Run the following command from the project root:

```powershell
python Inference.py
```

Example inference GUI:

<div align="center">
  <img width="760" src="docs/images/Inference.png" />
</div>

In the inference GUI, configure:

- `Mode`: `TIM` or `SDCM`
- `Organelle`
- `TIM` defaults to `XY = 59 nm`, `Z = 59 nm`, with only `Super-resolution` checked
- `SDCM` defaults to `Denoise + Super-resolution`
- `SDCM Tub` and `Rab7` default to `XY = 65 nm`, `Z = 200 nm`
- `SDCM Tomm20` defaults to `XY = 65 nm`, `Z = 300 nm`
- input TIFF folder
- `XY` and `Z` voxel size in nanometers if you want to override the preset
- denoise-only output keeps the input voxel size
- SR output is fixed to `6x` in `X / Y / Z`

## Command-Line Examples

### Denoise Training

```powershell
python Train.py `
  --mode TIM `
  --organelle Tub `
  --task Denoise `
  --hr-path \path\to\Denoise_data\HSNR `
  --lr-path \path\to\Denoise_data\LSNR
```

### SR Training

```powershell
python Train.py `
  --mode SDCM `
  --organelle Tub `
  --task SR `
  --hr-path \path\to\SR_data\GT `
  --lr-path \path\to\SR_data\LR
```

### Inference

```powershell
python Inference.py `
  --mode SDCM `
  --organelle Tub `
  --task both `
  --input-path \path\data_dir `
  --xy-nm 65 `
  --z-nm 200
```

## Notes

- For inference, the required pretrained models must exist under `experiments/`.
- If the required inference arguments are not provided, the Python GUI opens automatically.
- Pretrained models should be placed under `HBExMNet/experiments`.
- The standalone Windows C++ inference software is released separately for users who want direct desktop deployment without configuring Python.

## Acknowledgements

This program was developed using deep learning via PyTorch. We also acknowledge the generous contributions of Xintao Wang et al.[1] and Martin Weigert et al.[2]. You are welcome to use the code or program freely for research purposes. For further inquiries, please contact us at feipeng@hust.edu.cn or chenlongbiao@hust.edu.cn.

## References
1. Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy, and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2022.
2. Martin Weigert, Uwe Schmidt, Tobias Boothe, Andreas Muller, Alexandr Dibrov, Akanksha Jain, Benjamin Wilhelm, Deborah Schmidt, Coleman Broaddus, and Gene Myers. Content-aware image restoration: pushing the limits of fluorescence microscopy. Nature Methods, 15, 1090-1097, 2018.

