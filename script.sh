python train.py --dataset_name 'exchangerate'

python infer.py --dataset_name 'exchangerate_24' --cfg_scale 7.0 --total_step 100
python infer.py --dataset_name 'exchangerate_48' --cfg_scale 12.0 --total_step 60
python infer.py --dataset_name 'exchangerate_96' --cfg_scale 5.0 --total_step 100

python evaluation.py --dataset_name 'exchangerate_24' --cfg_scale 7.0 --total_step 100  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'exchangerate_48' --cfg_scale 12.0 --total_step 60  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'exchangerate_96' --cfg_scale 5.0 --total_step 100  # If you want to verify the experimental results, please run this line.



python train.py --dataset_name 'electricity'

python infer.py --dataset_name 'electricity_24' --cfg_scale 5.0 --total_step 60
python infer.py --dataset_name 'electricity_48' --cfg_scale 5.0 --total_step 10
python infer.py --dataset_name 'electricity_96' --cfg_scale 13.0 --total_step 30

python evaluation.py --dataset_name 'electricity_24' --cfg_scale 5.0 --total_step 60  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'electricity_48' --cfg_scale 5.0 --total_step 10  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'electricity_96' --cfg_scale 13.0 --total_step 30  # If you want to verify the experimental results, please run this line.



python train.py --dataset_name 'traffic'

python infer.py --dataset_name 'traffic_24' --cfg_scale 5.0 --total_step 100
python infer.py --dataset_name 'traffic_48' --cfg_scale 5.0 --total_step 10
python infer.py --dataset_name 'traffic_96' --cfg_scale 5.0 --total_step 30

python evaluation.py --dataset_name 'traffic_24' --cfg_scale 5.0 --total_step 100  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'traffic_48' --cfg_scale 5.0 --total_step 10  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'traffic_96' --cfg_scale 5.0 --total_step 30  # If you want to verify the experimental results, please run this line.



python train.py --dataset_name 'ETTh1'

python infer.py --dataset_name 'ETTh1_24' --cfg_scale 9.0 --total_step 10
python infer.py --dataset_name 'ETTh1_48' --cfg_scale 9.0 --total_step 10
python infer.py --dataset_name 'ETTh1_96' --cfg_scale 9.0 --total_step 10

python evaluation.py --dataset_name 'ETTh1_24' --cfg_scale 9.0 --total_step 10  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'ETTh1_48' --cfg_scale 9.0 --total_step 10  # If you want to verify the experimental results, please run this line.
python evaluation.py --dataset_name 'ETTh1_96' --cfg_scale 9.0 --total_step 10  # If you want to verify the experimental results, please run this line.




