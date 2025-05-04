import pandas as pd
import numpy as np
import os
import ast

###########################################
#         read candidate csv data        #
###########################################
domain = ['ETTh1', 'ETTm1', 'electricity', 'exchange_rate', 'air-quality','traffic']
domain_index = 0
length = 24
df = pd.read_csv(os.path.join('./Data/ours', f'embedding_cleaned_{domain[domain_index]}_{length}.csv'), index_col=False)


def cosine_similarity(a, b):
    a = np.array(a).ravel()
    b = np.array(b).ravel()
    dot_product = np.sum(a * b, axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = dot_product / (norm_a * norm_b)
    similarity = np.nan_to_num(similarity)
    return similarity


def clean_embedding_string(embedding_str):
    return [float(num) for num in embedding_str.replace('[', '').replace(']', '').strip().split()]


# malloc space and get unique sample id
final_rows = []
unique_sample_ids = df['SampleID'].unique()

for sample_id in unique_sample_ids:
    sample_groups = df[df['SampleID'] == sample_id]
    processed_embedding_samples = []
    print('Processing sample', sample_id)
    for index, sample_group in sample_groups.iterrows():
        embedding_sample = clean_embedding_string(sample_group['TextEmbedding'])
        processed_embedding_samples.append(embedding_sample)

    embeddings = np.array(processed_embedding_samples)

    # calculate cosine similarity
    similarity_matrix = np.array(
        [[cosine_similarity(embeddings[i], embeddings[j]) for j in range(len(embeddings))] for i in
         range(len(embeddings))]
    )
    similarity_sums = similarity_matrix.sum(axis=1)
    max_index = np.argmax(similarity_sums)

    sample_group = df[(df['SampleID'] == sample_id) & (df['SampleNumID'] == max_index + 1)]
    final_rows.append(sample_group)

final_df = pd.concat(final_rows, ignore_index=True)
final_df['SampleNumID'] = 1
final_df.to_csv(os.path.join('./Data/ours', f'embedding_cleaned_{domain[domain_index]}_{length}_refined.csv'),
                index=False)
print("Refined CSV file saved successfully.")