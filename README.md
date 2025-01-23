<div align="center">
  <!-- <h1><b> T2S </b></h1> -->
  <!-- <h2><b> T2S </b></h2> -->
  <h2><b> (IJCAI'25) T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models </b></h2>
</div>


## Requirements

Use python 3.11 from Conda

- timm

To install all dependencies:

```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets by unzip Data.zip and move under `./Data`.

## Checkpoints
You can access the well checkpoints by unzip results.zip and move under `./results`.

## test scripts
'''

python train.py --dataset_name 'exchangerate'

python infer.py --dataset_name 'exchangerate_24' --cfg_scale 7 --total_step 100
python infer.py --dataset_name 'exchangerate_48' --cfg_scale 12 --total_step 60
python infer.py --dataset_name 'exchangerate_96' --cfg_scale 5 --total_step 100

python evaluation.py --dataset_name 'exchangerate_24' --cfg_scale 7 --total_step 100
python evaluation.py --dataset_name 'exchangerate_48' --cfg_scale 12 --total_step 60
python evaluation.py --dataset_name 'exchangerate_96' --cfg_scale 5 --total_step 100





python train.py --dataset_name 'electricity'

python infer.py --dataset_name 'electricity_24' --cfg_scale 5.0 --total_step 60
python infer.py --dataset_name 'electricity_48' --cfg_scale 5.0 --total_step 10
python infer.py --dataset_name 'electricity_96' --cfg_scale 13.0 --total_step 30

python evaluation.py --dataset_name 'electricity_24' --cfg_scale 5.0 --total_step 60
python evaluation.py --dataset_name 'electricity_48' --cfg_scale 5.0 --total_step 10
python evaluation.py --dataset_name 'electricity_96' --cfg_scale 13.0 --total_step 30




python train.py --dataset_name 'traffic'

python infer.py --dataset_name 'traffic_24' --cfg_scale 5.0 --total_step 100
python infer.py --dataset_name 'traffic_48' --cfg_scale 5.0 --total_step 10
python infer.py --dataset_name 'traffic_96' --cfg_scale 5.0 --total_step 30

python evaluation.py --dataset_name 'traffic_24' --cfg_scale 5.0 --total_step 100
python evaluation.py --dataset_name 'traffic_48' --cfg_scale 5.0 --total_step 10
python evaluation.py --dataset_name 'traffic_96' --cfg_scale 5.0 --total_step 30




python train.py --dataset_name 'ETTh1'

python infer.py --dataset_name 'ETTh1_24' --cfg_scale 9.0 --total_step 10
python infer.py --dataset_name 'ETTh1_48' --cfg_scale 9.0 --total_step 10
python infer.py --dataset_name 'ETTh1_96' --cfg_scale 9.0 --total_step 10

python evaluation.py --dataset_name 'ETTh1_24' --cfg_scale 9.0 --total_step 10
python evaluation.py --dataset_name 'ETTh1_48' --cfg_scale 9.0 --total_step 10
python evaluation.py --dataset_name 'ETTh1_96' --cfg_scale 9.0 --total_step 10



'''
