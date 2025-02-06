import os, json
import torch
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class T2SDataset(Dataset):
    def __init__(
            self,
            name='Agriculture',
            data_root='./Data/MMD',
            window=24,
            proportion=0.99,
            seed=123,
            period='train',
            max_length=32,
    ):
        super(T2SDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        self.window, self.period, self.name,self.max_length = window, period, name, max_length
        self.csv_data = pd.read_csv(os.path.join(data_root, name+'.csv'))
        self.data, self.text, self.embedding = self.read_data(self.name, self.csv_data)
        self.len, self.var_num = self.data.shape[-1], 1
        train_data, inference_data, train_text, test_text, train_embedding, test_embedding = self.__getsamples(self.data, self.text, self.embedding, proportion, seed )
        self.samples = train_data if period == 'train' else inference_data
        self.text = train_text if period == 'train' else test_text
        self.embedding = train_embedding if period == 'train' else test_embedding
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, text, embedding, proportion, seed):
        train_data, test_data, train_text, test_text, train_embedding, test_embedding= self.divide(data, text, embedding, proportion, seed)
        return train_data, test_data, train_text, test_text, train_embedding, test_embedding

    def __normalize(self, rawdata):
        return self.scaler.transform(rawdata)

    def __unnormalize(self, data):
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(data, text, embedding, ratio, seed=2023):
        if not (data.shape[0] == len(text)):
            raise ValueError("All inputs must have the same number of rows.")
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]
        # splited TSdata and Text
        if data.ndim == 1:
            regular_data = data[regular_train_id]
            irregular_data = data[irregular_train_id]
        else:
            regular_data = data[regular_train_id, :]
            irregular_data = data[irregular_train_id, :]

        regular_text = [text[i] for i in regular_train_id]
        irregular_text = [text[i] for i in irregular_train_id]
        regular_embedding = embedding[regular_train_id, :]
        irregular_embedding = embedding[irregular_train_id, :]
        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data, regular_text, irregular_text, regular_embedding, irregular_embedding

    @staticmethod
    def read_data(name, json_root_data):
        trend_texts = json_root_data['Text'].tolist()
        text = trend_texts

        parsed_data = [ast.literal_eval(item) if isinstance(item, str) else item for item in json_root_data['OT']]
        if 'TSL' in name.split('_'):
            time_series = np.array(parsed_data, dtype=object)
        else:
            time_series = np.array(parsed_data)
            scaler = MinMaxScaler().fit(time_series)
            time_series = scaler.transform(time_series)

        if any(part in ['Agriculture','Climate','Energy','Health','Security','Traffic','Economy','Environment','SocialGood', 'SUSHI'] for part in name.split('_')):
            list_of_arrays = json_root_data['TextEmbedding'].apply(
                lambda embedding_str: np.array(ast.literal_eval(embedding_str))
            )
            embedding = np.array(list_of_arrays.tolist())
        else:
            list_of_arrays = json_root_data['TextEmbedding'].apply(
                lambda embedding_str: [float(num) for num in
                                       embedding_str.replace('[', '').replace(']', '').strip().split()]
            )
            embedding = np.array(list_of_arrays.tolist())

        return time_series, text, embedding

    def __getitem__(self, ind):
        x = self.samples[ind]
        text = self.text[ind]
        embedding = self.embedding[ind]
        return text, x, embedding

    def __len__(self):
        return self.sample_num