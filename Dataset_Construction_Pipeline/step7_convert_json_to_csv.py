import pandas as pd
import numpy as np
import os
import json


domain = ['ETTh1', 'ETTm1', 'electricity', 'exchange_rate', 'air-quality', 'traffic']
domain_index = 0
length = 96
data_root = f'./json_data_{domain[domain_index]}_1029_small_embedding_{length}/'
json_files = sorted([os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.json')])
json_data = []


for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data.append(json.load(f))


trend_texts = [sample.get('Trend Analysis', '') for sample in json_data]
time_series = np.array([sample.get('sampled_time_series', '') for sample in json_data])
embedding = np.array([sample.get('embedding', '') for sample in json_data])


new_data = {
    'SampleID': [],
    'SampleNumID': [],
    'TimeInterval': [],
    'Text': [],
    'TextEmbedding': [],
    'OT': []
}


for index, sample in enumerate(json_data):
    file_name = os.path.basename(json_files[index])
    parts = file_name.split('_')  
    if len(parts) >= 4:  
        start_date = parts[2]  
        end_date = parts[3].replace('.json', '')  
    else:
        print(f"Warning: file name {file_name} does not have the expected format. Skipping this entry.")
        continue
    text_combined = trend_texts[index]  
    ot_list = list(time_series[index])  
    if not isinstance(text_combined, str):
        print(f"Warning: text_combined at index {index} is not a string. Skipping this entry.")
        continue
    if not isinstance(ot_list, list):
        print(f"Warning: ot_list at index {index} is not a list. Skipping this entry.")
        continue
    if len(ot_list) < length:
        last_value = ot_list[-1] if ot_list else 0  
        ot_list.extend([last_value] * (length - len(ot_list)))
        print(f"ot_list at index {index} was extended to length {length}.")
    elif len(ot_list) > length:
        print(f"Warning: ot_list at index {index} is longer than {length}. Skipping this entry.")
        continue
    if not (isinstance(embedding[index], (list, np.ndarray)) and len(embedding[index]) == 128):
        print(f"Warning: embedding at index {index} is not a list or ndarray of length 128. Skipping this entry.")
        continue
    new_data['SampleID'].append(start_date)
    new_data['SampleNumID'].append(end_date)
    new_data['TimeInterval'].append(length)
    new_data['Text'].append(text_combined)
    new_data['OT'].append(ot_list)
    new_data['TextEmbedding'].append(embedding[index])

new_df = pd.DataFrame(new_data)
new_df.to_csv(os.path.join('./Data/ours', f'embedding_cleaned_{domain[domain_index]}_{length}.csv'), index=False)
print("CSV file aggregated successfully")
