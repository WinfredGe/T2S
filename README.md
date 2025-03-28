<div align="center">
  <!-- <h1><b> T2S </b></h1> -->
  <!-- <h2><b> T2S </b></h2> -->
  <h2><b> (IJCAI'25) T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models </b></h2>
</div>


## Requirements

Use python 3.10 from Conda

- torch==2.3.1
- datasets==2.21.0
- einops==0.7.0
- numpy==1.26.4
- pandas==1.5.3
- scipy==1.14.1
- seaborn==0.13.2
- tqdm==4.66.5
- transformers==4.47.0
- timm==1.0.11
- sentencepiece==0.2.0
- peft==0.10.0
- openai==1.35.9

To install all dependencies:

```shell
pip install -r requirements.txt
```

## Get Started

1. Install Python 3.10 and PyTorch 2.3.1.
2. Download the [data](https://drive.google.com/file/d/1sAZ8VNain_q4-LegtpRNCh83-dhO1jtm/view?usp=sharing) and [checkpoints](https://drive.google.com/file/d/1ghiepbJFS6DlK9gvgq1dvRla9cgQz56g/view?usp=sharing) from Google Drive to `./`
3. Train and perform inference with the model. We provide the experiment script under the  `./script.sh`. (optional)
4. You can evaluate the model using  `./scripts_validation_only.sh` or by:

```shell
python evaluation.py --dataset_name 'exchangerate_24' --cfg_scale 7.0 --total_step 100
python evaluation.py --dataset_name 'exchangerate_48' --cfg_scale 12.0 --total_step 60
python evaluation.py --dataset_name 'exchangerate_96' --cfg_scale 5.0 --total_step 100

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
