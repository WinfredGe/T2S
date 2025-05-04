import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
from evaluate.ts2vec import initialize_ts2vec
from evaluate.feature_based_measures import calculate_mdd, calculate_acd, calculate_sd, calculate_kd
import os
import datetime
from evaluate.utils import show_with_start_divider, show_with_end_divider, determine_device, write_json_data
import argparse
import torch
from scipy.stats import norm

###################################################
#                    MRR                          #
###################################################

def calculate_mrr(ori_data, gen_data, k=None):
    n_batch_size = ori_data.shape[0]
    n_generations = gen_data.shape[3]
    k = n_generations if k is None else k

    mrr_scores = np.zeros(n_batch_size)

    for batch_idx in range(n_batch_size):
        similarities = []
        for gen_idx in range(k):
            real_sequence = ori_data[batch_idx]
            generated_sequence = gen_data[batch_idx, :, :, gen_idx]
            similarity = cosine_similarity(real_sequence, generated_sequence)
            similarities.append(np.mean(similarity))

        sorted_indices = np.argsort(similarities)[::-1]
        rank = None
        for idx in sorted_indices:
            if similarities[idx] > therehold:
                rank = idx + 1
                break

        mrr_scores[batch_idx] = 1.0 / rank if rank is not None else 0.0

    return np.mean(mrr_scores)

###################################################
#             other reconstruct:CRPS              #
###################################################

def calculate_crps(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_timesteps = ori_data.shape[1]
    n_series = ori_data.shape[2]
    n_generations = gen_data.shape[3]
    crps_values = []

    for i in range(n_samples):
        total_crps = 0

        for j in range(n_series):
            crps_list = []

            for k in range(n_generations):
                mean = gen_data[i, :, j, k].mean()
                std_dev = gen_data[i, :, j, k].std()
                if std_dev == 0:
                    std_dev += 1e-8
                obs_value = ori_data[i, :, j]
                cdf_obs = np.where(obs_value < mean, 0, 1)

                cdf_pred = norm.cdf(obs_value, loc=mean, scale=std_dev)

                crps = np.mean((cdf_obs - cdf_pred) ** 2)
                crps_list.append(crps)

            average_crps = np.mean(crps_list)
            total_crps += average_crps

        crps_values.append(total_crps / n_series)

    crps_values = np.array(crps_values)
    average_crps = crps_values.mean()
    return average_crps


def evaluate_muldata(args, ori_data, gen_data):
    show_with_start_divider(f"Evalution with settings:{args}")

    # Parse configs
    method_list = args.method_list
    dataset_name = args.dataset_name
    model_name = args.model_name
    device = args.device
    evaluation_save_path = args.evaluation_save_path

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M%S")
    combined_name = f'{model_name}_{dataset_name}_{formatted_time}_multi'

    if not isinstance(method_list, list):
        method_list = method_list.strip('[]')
        method_list = [method.strip() for method in method_list.split(',')]
    if gen_data is None:
        show_with_end_divider('Error: Generated data not found.')
        return None

    result = {}

    if 'CRPS' in method_list:
        mdd = calculate_crps(ori_data, gen_data)
        result['CRPS'] = mdd
    if 'MRR' in method_list:
        mrr = calculate_mrr(ori_data, gen_data)
        result['MRR'] = mrr

    if isinstance(result, dict):
        evaluation_save_path = os.path.join(evaluation_save_path, f'{combined_name}.json')
        write_json_data(result, evaluation_save_path)
        print(f'Evaluation denoiser_results saved to {evaluation_save_path}.')

    show_with_end_divider(f'Evaluation done. Results:{result}.')

    return result


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_ed(ori_data,gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0
        for j in range(n_series):
            distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_distance_eu += distance
        distance_eu.append(total_distance_eu / n_series)

    distance_eu = np.array(distance_eu)
    average_distance_eu = distance_eu.mean()
    return average_distance_eu

def calculate_dtw(ori_data,comp_data):
    distance_dtw = []
    n_samples = ori_data.shape[0]
    for i in range(n_samples):
        distance = multi_dtw_distance(ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True)
        distance_dtw.append(distance)

    distance_dtw = np.array(distance_dtw)
    average_distance_dtw = distance_dtw.mean()
    return average_distance_dtw

import numpy as np


def calculate_mse(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    mse_values = []

    for i in range(n_samples):
        total_mse = 0
        for j in range(n_series):
            mse = np.mean((ori_data[i, :, j] - gen_data[i, :, j]) ** 2)
            total_mse += mse
        mse_values.append(total_mse / n_series)

    mse_values = np.array(mse_values)
    average_mse = mse_values.mean()
    return average_mse


def calculate_wape(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    wape_values = []

    for i in range(n_samples):
        total_absolute_error = 0
        total_actual_value = 0

        for j in range(n_series):
            absolute_error = np.abs(ori_data[i, :, j] - gen_data[i, :, j])
            total_absolute_error += np.sum(absolute_error)
            total_actual_value += np.sum(np.abs(ori_data[i, :, j]))

        if total_actual_value != 0:
            wape = total_absolute_error / total_actual_value
        else:
            wape = np.nan

        wape_values.append(wape)

    wape_values = np.array(wape_values)
    average_wape = np.nanmean(wape_values)
    return average_wape



def evaluate_data(args, ori_data, gen_data):
    show_with_start_divider(f"Evalution with settings:{args}")

    # Parse configs
    method_list = args.method_list
    dataset_name = args.dataset_name
    model_name = args.model_name
    device = args.device
    evaluation_save_path = args.evaluation_save_path

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M%S")
    combined_name = f'{model_name}_{dataset_name}_{formatted_time}'

    if not isinstance(method_list,list):
        method_list = method_list.strip('[]')
        method_list = [method.strip() for method in method_list.split(',')]

    if gen_data is None:
        show_with_end_divider('Error: Generated data not found.')
        return None
    if ori_data.shape != gen_data.shape:
        print(f'Original data shape: {ori_data.shape}, Generated data shape: {gen_data.shape}.')
        show_with_end_divider('Error: Generated data does not have the same shape with original data.')
        return None
    
    result = {}

    if 'C-FID' in method_list:
        fid_model = initialize_ts2vec(np.transpose(ori_data, (0, 2, 1)),device)
        ori_repr = fid_model.encode(np.transpose(ori_data,(0, 2, 1)), encoding_window='full_series')
        gen_repr = fid_model.encode(np.transpose(gen_data,(0, 2, 1)), encoding_window='full_series')
        cfid = calculate_fid(ori_repr,gen_repr)
        result['C-FID'] = cfid

    ori_data = np.transpose(ori_data, (0, 2, 1))
    gen_data = np.transpose(gen_data, (0, 2, 1))

    if 'MSE' in method_list:
        mse = calculate_mse(ori_data,gen_data)
        result['MSE'] = mse
    if 'WAPE' in method_list:
        wape = calculate_wape(ori_data,gen_data)
        result['WAPE'] = wape


    ori_data = np.transpose(ori_data, (0, 2, 1))
    gen_data = np.transpose(gen_data, (0, 2, 1))

    if isinstance(result, dict):
        evaluation_save_path = os.path.join(evaluation_save_path, f'{combined_name}.json')
        write_json_data(result, evaluation_save_path)
        print(f'Evaluation denoiser_results saved to {evaluation_save_path}.')
    
    show_with_end_divider(f'Evaluation done. Results:{result}.')

    return result

parser = argparse.ArgumentParser(description="Train flow matching model")
parser.add_argument('--method_list', type=str, default='MSE,WAPE,MRR',
                        help='metric list [MSE,WAPE,MRR]')
parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Denoiser Model save path')
parser.add_argument('--dataset_name', type=str, default='ETTh1_96', help='dataset name')
parser.add_argument('--backbone', type=str, default='flowmatching', help='flowmatching or DDPM or EDM')
parser.add_argument('--denoiser', type=str, default='DiT', help='DiT or MLP')
# parser.add_argument('--data_length', type=int, default=96, help='24 48 96')
parser.add_argument('--cfg_scale', type=float, default=9.0, help='CFG Scale')
parser.add_argument('--total_step', type=int, default=10, help='total step sampled from [0,1]')

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.data_length = args.dataset_name.split('_')[-1] if args.dataset_name != 'SUSHI' else 2048
args.model_name = '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name,args.cfg_scale, args.total_step)
# args.generation_save_path = os.path.join(args.save_path, 'generation', '{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name))
args.generation_save_path = os.path.join(args.save_path, 'generation',
                                         '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name,
                                                                 args.cfg_scale, args.total_step))
args.evaluation_save_path = os.path.join(args.save_path, 'evaluation', args.model_name)


'''evaluate our model'''
x_1 = np.load(os.path.join(args.generation_save_path,'run_0','x_1.npy'))
x_t = np.load(os.path.join(args.generation_save_path, 'x_t.npy'))
x_t_latent_dec_array = np.load(os.path.join(args.generation_save_path,'run_0','x_t_latent_dec_array.npy'))
x_t_latent_enc_array = np.load(os.path.join(args.generation_save_path, 'run_0','x_t_latent_enc_array.npy'))
x_1 = np.transpose(x_1, (0, 2, 1))
x_t = np.transpose(x_t, (0, 2, 1))
# print(f'x_1 shape:{x_1.shape}')
# print(f'x_t shape:{x_t.shape}')
evaluate_data(args, ori_data=x_1, gen_data=x_t)  # batch, dim , time length

therehold = 0.5
all_x_t = []
for run_index in range(10):
    '''Choice 1 : evaluate our model'''
    args.generation_save_path_result = os.path.join(args.generation_save_path, f'run_{run_index}')
    x_1 = np.load(os.path.join(args.generation_save_path_result, 'x_1.npy'))
    x_t = np.load(os.path.join(args.generation_save_path_result, 'x_t.npy'))

    x_t_expanded = np.expand_dims(x_t, axis=-1)
    all_x_t.append(x_t_expanded)

x_t_all = np.concatenate(all_x_t, axis=-1)
evaluate_muldata(args, ori_data=x_1, gen_data=x_t_all)
