import pandas as pd
import numpy as np
import os
import json
import os
import time
import json
from openai import OpenAI

os.environ['OPENAI_BASE_URL'] = "https://xxx.ADDyourGPTapi.com"
os.environ['OPENAI_API_KEY'] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
client = OpenAI()


def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model, dimensions=128).data[0].embedding  # 128 is fixed


def process_json_files(source_directory, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created target directory: {target_directory}")

    for filename in os.listdir(source_directory):
        if filename.endswith('.json'):
            source_file_path = os.path.join(source_directory, filename)
            target_file_path = os.path.join(target_directory, filename)
            if os.path.exists(target_file_path):
                print(f"File {target_file_path} already exists. Skipping.")
                continue

            print(f"Processing file: {source_file_path}")

            try:
                with open(source_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                trend_analysis = data.get("Trend Analysis", "")
                if not trend_analysis:
                    print(f"No 'Trend Analysis' field found in {filename}. Skipping.")
                    continue

                start_time = time.time()
                embedding = get_embedding(trend_analysis)
                end_time = time.time()
                if embedding is None:
                    print(f"Failed to get embedding for {filename}. Skipping.")
                    continue

                data["embedding"] = embedding

                with open(target_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                print(
                    f"Successfully updated and saved {filename} to {target_directory}.cost time:{end_time - start_time}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")


def check_json_format(file_path,sample_length):
    """Check if the JSON file format meets the requirements"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            sampled_time_series = data.get("sampled_time_series", [])
            if not isinstance(sampled_time_series, list) or len(sampled_time_series) != sample_length:
                return False, f"The 'sampled_time_series' field must be a list of length {sample_length}."

            embedding = data.get("embedding", [])
            if not isinstance(embedding, list) or len(embedding) != 128:
                return False, "The 'embedding' field must be a list of length 128."

            return True, "Valid JSON format."

    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"An error occurred: {e}"


def check_json_files_in_directory(directory,sample_length):
    invalid_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            is_valid, message = check_json_format(file_path,sample_length)
            if not is_valid:
                invalid_files.append(f"File '{filename}' has issues: {message}")

    return invalid_files

###############################################################
#                        json_get_embedding                   #
###############################################################
dataset_name_index = 1
sample_length_index = 0
dataset_name_list = ['ETTh1', 'ETTm1', 'electricity', 'exchange_rate', 'traffic', 'air-quality']
dataset_name = dataset_name_list[dataset_name_index]
target_column_name = 'OT'
sample_length_list = [24, 48, 96]  #
sample_length = sample_length_list[sample_length_index]
source_directory = f'./{dataset_name}_{sample_length}_folder/json_data'
target_directory = f'./json_data_{dataset_name}_1029_small_embedding_{sample_length}'

process_json_files(source_directory, target_directory)



###############################################################
#                  check_content_validation                   #
###############################################################

dataset_name_index = 1
sample_length_index = 2
dataset_name_list = ['ETTh1', 'ETTm1', 'electricity', 'exchange_rate', 'national_illness', 'traffic',
                     'weather', 'm4']
dataset_name = dataset_name_list[dataset_name_index]
target_column_name = 'OT'
sample_length_list = [24, 48, 96]
sample_length = sample_length_list[sample_length_index]
directory_path = f'./json_data_{dataset_name}_1029_small_embedding_{sample_length}'

invalid_files = check_json_files_in_directory(directory_path,sample_length)

if invalid_files:
    for issue in invalid_files:
        print(issue)
else:
    print("All files are valid.")

###############################################################
#                       convert_json_to_csv                   #
###############################################################

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
