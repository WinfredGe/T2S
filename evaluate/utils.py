import os
import pickle
import json
import torch
import numpy as np

PREPROCESSING_PARAS = ['do_preprocessing','original_data_path','output_ori_path','dataset_name','use_ucr_uea_dataset','ucr_uea_dataset_name','seq_length','valid_ratio','do_normalization']
GENERATION_PARAS = ['do_generation','model','dataset_name']

def show_divider():
    print("=" * 20)

def show_with_start_divider(content):
    show_divider()
    print(content)

def show_with_end_divider(content):
    print(content)
    show_divider()
    print()

def make_sure_path_exist(path):
    if os.path.isdir(path) and not path.endswith(os.sep):
        dir_path = path
    else:
        dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)


def write_json_data(content, path):
    make_sure_path_exist(path)
    with open(path, 'w') as json_file:
        json.dump(content, json_file, indent=4)

def determine_device(no_cuda,cuda_device):
    # Determine device (cpu/gpu)
    if no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if torch.cuda.device_count()>1:
            device = torch.device('cuda', cuda_device)
        else:
            device = torch.device('cuda', 0)
    return device

class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data

