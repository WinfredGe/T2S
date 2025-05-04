import os
import json


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

