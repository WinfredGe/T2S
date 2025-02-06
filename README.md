<div align="center">
  <!-- <h1><b> T2S </b></h1> -->
  <!-- <h2><b> T2S </b></h2> -->
  <h2><b> (IJCAI'25) T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models </b></h2>
</div>


## Requirements

Use python 3.10 from Conda

- timm



## Get Started

1. Install Python 3.10 and PyTorch 2.3.1.
2. Download the data and checkpoints from [Google Drive](https://drive.google.com/drive/folders/1W2GyFTeDiS5te3PhYJqtZmcCvU4blgoG?usp=sharing) to `./`
3. Train and perform inference with the model. We provide the experiment script under the  `./script.sh`. (optional)
4. You can evaluate the model using  `./scripts_validation_only.sh` or by:

```shell
python evaluation.py --dataset_name 'exchangerate_24' --cfg_scale 7 --total_step 100
python evaluation.py --dataset_name 'exchangerate_48' --cfg_scale 12 --total_step 60
python evaluation.py --dataset_name 'exchangerate_96' --cfg_scale 5 --total_step 100

python evaluation.py --dataset_name 'electricity_24' --cfg_scale 5.0 --total_step 60
python evaluation.py --dataset_name 'electricity_48' --cfg_scale 5.0 --total_step 10
python evaluation.py --dataset_name 'electricity_96' --cfg_scale 13.0 --total_step 30


python evaluation.py --dataset_name 'traffic_24' --cfg_scale 5.0 --total_step 100
python evaluation.py --dataset_name 'traffic_48' --cfg_scale 5.0 --total_step 10
python evaluation.py --dataset_name 'traffic_96' --cfg_scale 5.0 --total_step 30

python evaluation.py --dataset_name 'ETTh1_24' --cfg_scale 9.0 --total_step 10
python evaluation.py --dataset_name 'ETTh1_48' --cfg_scale 9.0 --total_step 10
python evaluation.py --dataset_name 'ETTh1_96' --cfg_scale 9.0 --total_step 10
```

`
