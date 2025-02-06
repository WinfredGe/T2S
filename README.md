<div align="center">
  <!-- <h1><b> T2S </b></h1> -->
  <!-- <h2><b> T2S </b></h2> -->
  <h2><b> (IJCAI'25) T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models </b></h2>
</div>


## Requirements

Use python 3.11 from Conda

- timm


## test scripts
'''

python train.py --dataset_name 'exchangerate'

python infer.py --dataset_name 'exchangerate_24' --cfg_scale 7 --total_step 100
python infer.py --dataset_name 'exchangerate_48' --cfg_scale 12 --total_step 60
python infer.py --dataset_name 'exchangerate_96' --cfg_scale 5 --total_step 100

python evaluation.py --dataset_name 'exchangerate_24' --cfg_scale 7 --total_step 100
python evaluation.py --dataset_name 'exchangerate_48' --cfg_scale 12 --total_step 60
python evaluation.py --dataset_name 'exchangerate_96' --cfg_scale 5 --total_step 100

'''
