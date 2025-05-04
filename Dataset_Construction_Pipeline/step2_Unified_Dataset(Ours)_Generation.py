import os
import openai
from matplotlib import pyplot as plt
import pandas as pd
import json
import time

os.environ['OPENAI_BASE_URL'] = "https://xxx.ADDyourGPTapi.com"
os.environ['OPENAI_API_KEY'] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
client = openai.Client()
###########################################################
#                      settings                           #
###########################################################
dataset_name_index = 0
sample_length_index = 0

dataset_name_list = ['ETTh1', 'ETTm1', 'electricity', 'exchange_rate', 'traffic', 'air-quality']
sample_length_list = [24, 48, 96]
dataset_name = dataset_name_list[dataset_name_index]
dataset_path = f'./{dataset_name}.csv'
target_column_name = 'OT'
#
sample_length = sample_length_list[sample_length_index]
saved_path = f'./{dataset_name}_{sample_length}_folder'
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
    print(f"Path '{saved_path}' created.")
else:
    print(f"Path '{saved_path}' already exists.")
data = pd.read_csv(dataset_path)
csv_data_file_total_length = len(data)
Max_Iteration = csv_data_file_total_length - sample_length
data = data[target_column_name]
data.index = pd.to_datetime(data.index)
list_time_data = data.to_list()

dataset = []
for i in range(Max_Iteration):
    sample = list_time_data[i: i + sample_length]
    dataset.append(sample)
print(len(dataset))
print(len(dataset[0]))


###########################################################
#                      function                           #
###########################################################

def get_completion(user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You're an expert in time series summarization, providing insightful and succinct descriptions with precise language. Avoid unnecessary text or explanations."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    return completion.choices[0].message.content




def plot_data_to_picture(data, text, sample_num):
    fig, ax = plt.subplots()
    ax.plot(data)
    fig.text(0.5, 0.05, text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.tight_layout()
    # if not picture file create
    directory = os.path.join(saved_path, 'picture')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(saved_path + '/picture/data_sample_{}.png'.format(sample_num), bbox_inches='tight')


def one_sample_data_summary(data):
    formatted_data = [f"{i + 1}.0, {value:.3f}" for i, value in enumerate(data)]
    formatted_string = "\n".join(formatted_data)
    formatted_json = """
      {
        "Trend Analysis": "..."
      }
      """
    user_prompt = f"""1.Summarize the observed trend in the given time series data.
      2.ONLY output the summary using the following JSON format.
      3.The output MUST be less than 256 tokens.
      4.The output description MUST be consistent with the actual trend characteristics of the time series.
      Given the time series data
      ```{formatted_string}```
      Use the following JSON format:
      ```{formatted_json}```
      """

    start_time = time.time()
    one_sample_completion = get_completion(user_prompt)
    end_time = time.time()
    # print(one_sample_completion)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    # encoding = tiktoken.encoding_for_model("gpt-4o")
    # print(f"Token count: {len(encoding.encode(user_prompt))}")
    return one_sample_completion


def save_data_to_json(sample, one_sample_completion, sample_num, itr_num):
    output_data = one_sample_completion
    start = output_data.find("`json") + 5
    end = output_data.find("`", start)
    json_string = output_data[start:end].strip()
    json_string_copy = json_string
    data = json.loads(json_string)
    # add new key
    data["sampled_time_series"] = sample
    # if not json_data file create
    directory = os.path.join(saved_path, 'json_data')
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(saved_path + '/json_data/data_sample_{}_{}.json'.format(sample_num, itr_num), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


###########################################################
#                         run                             #
###########################################################
max_retries = 3  # Set the maximum number of retries
start_index = 35223  # Start processing From xxx

progress_file = f'progress_{dataset_name}_{sample_length}.txt'
if os.path.exists(progress_file):
    with open(progress_file, 'r') as file:
        start_index = int(file.read().strip())
for i, sample in enumerate(dataset):
    if i < start_index:
        continue
    if i == len(dataset) - sample_length:
        break

    retries = 0
    while retries < max_retries:
        try:
            print('current_sample_is: {}'.format(i + 1))
            one_sample_text = one_sample_data_summary(sample)
            save_data_to_json(sample, one_sample_text, i + 1, 1)
            # one_sample_text = one_sample_data_summary(sample)
            # save_data_to_json(sample, one_sample_text, i + 1, 2)
            # one_sample_text = one_sample_data_summary(sample)
            # save_data_to_json(sample, one_sample_text, i + 1, 3)
            # one_sample_text = one_sample_data_summary(sample)
            # save_data_to_json(sample, one_sample_text, i + 1, 4)
            # one_sample_text = one_sample_data_summary(sample)
            # save_data_to_json(sample, one_sample_text, i + 1, 5)

            # plot_data_to_picture(sample, one_sample_text, i+1)

            with open(progress_file, 'w') as file:
                file.write(str(i + 1))

            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying {retries + 1}/{max_retries}...")
            retries += 1

    if retries == max_retries:
        error_message = f"Failed to process sample {i + 1} after {max_retries} retries."
        with open(f'error_log.txt_{dataset_name}_{sample_length}', 'a') as file:
            file.write(error_message + "\n")


