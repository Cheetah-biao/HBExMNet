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
|   |-- SDCM_Rab7_Denoise
|   |-- SDCM_Rab7_SR
|   |-- SDCM_Tomm20_Denoise
|   |-- SDCM_Tomm20_SR
|   |-- SDCM_Tub_Denoise
|   |-- SDCM_Tub_SR
|   |-- TIM_Rab7_Denoise
|   |-- TIM_Rab7_SR
|   |-- TIM_Tomm20_Denoise
|   |-- TIM_Tomm20_SR
|   |-- TIM_Tub_Denoise
|   `-- TIM_Tub_SR
|-- models
|-- options
`-- utils
```

In the final release:

- pretrained models under `experiments/` stay inside `HBExMNet`
- generated training patch files can be written into `HBExMNet/data`
- raw inference data and raw training data can stay outside the project folder

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

4. Download the pretrained models and datasets from the links below.

Recommended layout after extraction:

```text
base_dir\
|-- HBExMNet
|   |-- experiments
|   `-- data
|-- data
`-- Train_data
```

The GUI will look for `data` and `Train_data` beside the `HBExMNet` project folder by default.

## Model Zoo And Data Download

Large files are intentionally distributed outside the Git repository.

| Asset | Content | Extract To | Link |
| --- | --- | --- | --- |
| Pretrained models | trained checkpoints under `experiments/` | `HBExMNet/experiments/` | <div align="center"><a href="[https://drive.google.com/file/d/1-3rAfLrllMZDUWnQgUhWPFKFb2ahjIH3/view?usp=drive_link](https://drive.google.com/file/d/1HQ-yCvWVl-AANkWs_bIiMALu_q7TZKLA/view?usp=sharing)"><img src="https://img.shields.io/badge/Models-4285F4?style=flat-square&logo=googledrive&logoColor=white" alt="Models" /></a></div> |
| Test data package | example TIFF stacks for inference | sibling folder `data/` beside `HBExMNet/` | <div align="center"><a href="[https://drive.google.com/file/d/1-3rAfLrllMZDUWnQgUhWPFKFb2ahjIH3/view?usp=drive_link](https://drive.google.com/file/d/1xskC1NmFcO6JoV6DyWvYadZ482bAvckz/view?usp=sharing)"><img src="https://img.shields.io/badge/Data-4285F4?style=flat-square&logo=googledrive&logoColor=white" alt="Data" /></a></div> |
| Standalone Windows software | compiled C++ TensorRT inference package | any local folder | <div align="center"><a href="https://drive.google.com/file/d/1DJPh4Je8HOOB59zn1y3yAdo2cn21X7wj/view?usp=sharing"><img src="https://img.shields.io/badge/Software-4285F4?style=flat-square&logo=googledrive&logoColor=white" alt="Software" /></a></div> |

## Main Entry Points

Python training:

```text
Train.py
```

Python inference:

```text
Inference.py
```

GUI configuration files:

```text
options/train/Config_train.py
options/test/Config_inference.py
```

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

Outputs:

```text
data\training_data_denoise.npz
data\training_data_sr.npz
experiments\<MODE>_<Organelle>_<Task>
```

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

- `Mode`
- `Organelle`
- default workflow: keep only `Super-resolution` checked
- if you are processing live-cell data, also check `Denoise`
- input TIFF folder
- `XY` and `Z` voxel size in nanometers

Important behavior:

- `Z / XY` is computed automatically
- the default GUI voxel size is `XY = 65 nm`, `Z = 65 nm`
- denoise-only output keeps the input voxel size and does not apply interpolation or final downsampling
- SR output is fixed to `6x` in `X / Y / Z`, with no final downsampling
- output is written into a timestamped folder beside the selected input folder

## Command-Line Examples

### Denoise Training

```powershell
python Train.py `
  --mode TIM `
  --organelle Tub `
  --task Denoise `
  --hr-path \path\to\GT_or_MR `
  --lr-path \path\to\LR
```

Note:

- if `MR` exists beside `GT`, denoise training automatically prefers `MR` as the effective stage-1 supervision target

### SR Training

```powershell
python Train.py `
  --mode SDCM `
  --organelle Tub `
  --task SR `
  --hr-path \path\to\GT `
  --lr-path \path\to\LR
```

### Inference

```powershell
python Inference.py `
  --mode TIM `
  --organelle Tub `
  --task both `
  --input-path \path\to\data_dir `
  --xy-nm 65 `
  --z-nm 65
```

## Notes

- For inference, the required pretrained models must exist under `experiments/`.
- If the required inference arguments are not provided, the Python GUI opens automatically.
- For denoise training, the effective stage-1 supervision should match the LR scale.
- For SR training, only `LR` and `GT` are required. Intermediate supervision is generated implicitly by downsampling `GT` to the `LR` size during patch preparation.
- Pretrained models should be placed under `HBExMNet/experiments`.
- Raw inference data is recommended to stay in a sibling `data/` folder beside `HBExMNet`.
- Raw training data is recommended to stay in a sibling `Train_data/` folder beside `HBExMNet`.
- Generated patch files such as `training_data_denoise.npz` and `training_data_sr.npz` are written into `HBExMNet/data`.
- The standalone Windows C++ inference software is released separately for users who want direct desktop deployment without configuring Python.



