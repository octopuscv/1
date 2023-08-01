# SiamVIT: patchwise network for infrared small and dim objects location
<a href="#"><img src="https://img.shields.io/github/actions/workflow/status/milesial/PyTorch-UNet/main.yml?logo=github&style=for-the-badge" /></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.7.1+-red.svg?logo=PyTorch&style=for-the-badge" /></a> <a href="#"><img src="https://img.shields.io/badge/python-v3.9+-blue.svg?logo=python&style=for-the-badge" /></a>



- [DataSet](#quick-start)
- [Quick-start](#description)
  - [Training](#training)
  - [Detect](#prediction)
- [Pretrained model](#pretrained-model)
- [Data](#data)

## Dataset
We trained and tested our proposed SiamVIT using the public infrared [dataset](https://www.scidb.cn/en/detail?dataSetId=720626420933459968), which contains 22 sequences of approximately 16,000 images. we randomly selected 1,000 images for testing.
## Quick start
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.7.1 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Download the data

5. Run codes 
### Training
```bash
python trian.py
```
### Detect
```bash
python detect.py
```
## Pretrained model
Pre-trained weights `best_model_frame1_num_head8.pth` and `best_model_frame3_num_head8.pth` for these models are provided  [here](https://drive.google.com/drive/folders/1eDF10eWgL-w61E0iz0nfde37yixTY6ip?usp=sharing).

## ## Acknowledgement
Part of the code is borrowed from [VIT](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) and [GroupViT](https://github.com/NVlabs/GroupViT).(cheers to the community as well)
