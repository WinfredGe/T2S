from torch.utils.data import Dataset, DataLoader
from datafactory.dataset import T2SDataset
import torch
import numpy as np

class AlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.datasets = [dataset1, dataset2, dataset3]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.total_length = sum(self.lengths)
        self.index_map = {}
        offset = 0
        for i, length in enumerate(self.lengths):
            for j in range(length):
                self.index_map[offset + j] = (i, j)
            offset += length
    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        dataset_idx, sub_idx = self.index_map[index]
        return self.datasets[dataset_idx][sub_idx], dataset_idx



data_dict = {
    'ETTh1': 'embedding_cleaned_ETTh1',
    'ETTh1_24': 'embedding_cleaned_ETTh1_24',
    'ETTh1_48': 'embedding_cleaned_ETTh1_48',
    'ETTh1_96': 'embedding_cleaned_ETTh1_96',
    'ETTm1': 'embedding_cleaned_ETTm1',
    'ETTm1_24': 'embedding_cleaned_ETTm1_24',
    'ETTm1_48': 'embedding_cleaned_ETTm1_48',
    'ETTm1_96': 'embedding_cleaned_ETTm1_96',
    'airquality': 'embedding_cleaned_airquality',
    'airquality_24': 'embedding_cleaned_airquality_24',
    'airquality_48': 'embedding_cleaned_airquality_48',
    'airquality_96': 'embedding_cleaned_airquality_96',
    'electricity': 'embedding_cleaned_electricity',
    'electricity_24': 'embedding_cleaned_electricity_24',
    'electricity_48': 'embedding_cleaned_electricity_48',
    'electricity_96': 'embedding_cleaned_electricity_96',
    'exchangerate': 'embedding_cleaned_exchangerate',
    'exchangerate_24': 'embedding_cleaned_exchangerate_24',
    'exchangerate_48': 'embedding_cleaned_exchangerate_48',
    'exchangerate_96': 'embedding_cleaned_exchangerate_96',
    'nationalillness': 'embedding_cleaned_nationalillness',
    'nationalillness_24': 'embedding_cleaned_nationalillness_24',
    'nationalillness_48': 'embedding_cleaned_nationalillness_48',
    'nationalillness_96': 'embedding_cleaned_nationalillness_96',
    'traffic': 'embedding_cleaned_traffic',
    'traffic_24': 'embedding_cleaned_traffic_24',
    'traffic_48': 'embedding_cleaned_traffic_48',
    'traffic_96': 'embedding_cleaned_traffic_96',
    'weather': 'embedding_cleaned_weather',
    'weather_24': 'embedding_cleaned_weather_24',
    'weather_48': 'embedding_cleaned_weather_48',
    'weather_96': 'embedding_cleaned_weather_96',

    'MMD-Agriculture': 'embedding_cleaned_Agriculture',
    'MMD-Climate': 'embedding_cleaned_Climate',
    'MMD-Energy': 'embedding_cleaned_Energy',
    'MMD-Health_AFR': 'embedding_cleaned_Health_AFR',
    'MMD-Health_US': 'embedding_cleaned_Health_US',
    'MMD-Security': 'embedding_cleaned_Security',
    'MMD-Traffic': 'embedding_cleaned_Traffic',
    'MMD-Economy': 'embedding_cleaned_Economy',
    'MMD-Environment': 'embedding_cleaned_Environment',
    'MMD-SocialGood': 'embedding_cleaned_SocialGood',

    'MMD-Agriculture_24': 'embedding_cleaned_Agriculture_24',
    'MMD-Climate_24': 'embedding_cleaned_Climate_24',
    'MMD-Energy_24': 'embedding_cleaned_Energy_24',
    'MMD-Health_AFR_24': 'embedding_cleaned_Health_AFR_24',
    'MMD-Health_US_24': 'embedding_cleaned_Health_US_24',
    'MMD-Security_24': 'embedding_cleaned_Security_24',
    'MMD-Traffic_24': 'embedding_cleaned_Traffic_24',
    'MMD-Economy_24': 'embedding_cleaned_Economy_24',
    'MMD-Environment_24': 'embedding_cleaned_Environment_24',
    'MMD-SocialGood_24': 'embedding_cleaned_SocialGood_24',
    'MMD-Agriculture_48': 'embedding_cleaned_Agriculture_48',
    'MMD-Climate_48': 'embedding_cleaned_Climate_48',
    'MMD-Energy_48': 'embedding_cleaned_Energy_48',
    'MMD-Health_AFR_48': 'embedding_cleaned_Health_AFR_48',
    'MMD-Health_US_48': 'embedding_cleaned_Health_US_48',
    'MMD-Security_48': 'embedding_cleaned_Security_48',
    'MMD-Traffic_48': 'embedding_cleaned_Traffic_48',
    'MMD-Economy_48': 'embedding_cleaned_Economy_48',
    'MMD-Environment_48': 'embedding_cleaned_Environment_48',
    'MMD-SocialGood_48': 'embedding_cleaned_SocialGood_48',
    'MMD-Agriculture_96': 'embedding_cleaned_Agriculture_96',
    'MMD-Climate_96': 'embedding_cleaned_Climate_96',
    'MMD-Energy_96': 'embedding_cleaned_Energy_96',
    'MMD-Health_AFR_96': 'embedding_cleaned_Health_AFR_96',
    'MMD-Health_US_96': 'embedding_cleaned_Health_US_96',
    'MMD-Security_96': 'embedding_cleaned_Security_96',
    'MMD-Traffic_96': 'embedding_cleaned_Traffic_96',
    'MMD-Economy_96': 'embedding_cleaned_Economy_96',
    'MMD-Environment_96': 'embedding_cleaned_Environment_96',
    'MMD-SocialGood_96': 'embedding_cleaned_SocialGood_96',

    'TSL-Agriculture': 'embedding_Cleaned_TSL_AgricultureandFood',
    'TSL-Educational': 'embedding_Cleaned_TSL_EducationalandPublicServices',
    'TSL-Energy': 'embedding_Cleaned_TSL_EnergyandResourceConsumption',
    'TSL-Entertainment': 'embedding_Cleaned_TSL_Entertainment',
    'TSL-Environment': 'embedding_Cleaned_TSL_EnvironmentalandClimate',
    'TSL-Health': 'embedding_Cleaned_TSL_HealthandMedicalData',
    'TSL-Sales': 'embedding_Cleaned_TSL_SalesandMarketTrends',
    'TSL-Technological': 'embedding_Cleaned_TSL_TechnologicalandDigitalTrends',
    'TSL-Transport': 'embedding_Cleaned_TSL_TransportandTrafficTrends',
    'TSL-Wildlife': 'embedding_Cleaned_TSL_WildlifeandNature',

    'SUSHI': 'embedding_cleaned_SUSHI',
}
def loader_provider(args, period):
    if args.mix_train:
        Data_name = data_dict[args.dataset_name]
        if args.dataset_name in ['ETTh1', 'ETTm1','traffic', 'airquality', 'exchangerate', 'weather', 'electricity', 'nationalillness', 'weather']:
            dataset1 = T2SDataset(name=Data_name + '_24', data_root=f'./Data/our/', period=period)
            dataset2 = T2SDataset(name=Data_name + '_48', data_root=f'./Data/our/', period=period)
            dataset3 = T2SDataset(name=Data_name + '_96', data_root=f'./Data/our/', period=period)
            dataset = AlternatingDataset(dataset1, dataset2, dataset3)
        else:
            if args.dataset_name == 'SUSHI':
                Data_path = r"./Data/SUSHI/"
                dataset = T2SDataset(name=Data_name,data_root=Data_path, period=period)
            elif args.dataset_name.split('-')[0] == 'TSL':
                dataset = T2SDataset(name=Data_name,data_root=f'./Data/TSL/', period=period)
            elif args.dataset_name.split('-')[0] == 'MMD':
                dataset1 = T2SDataset(name=Data_name+'_24',data_root=f'./Data/MMD/', period=period)
                dataset2 = T2SDataset(name=Data_name+'_48',data_root=f'./Data/MMD/', period=period)
                dataset3 = T2SDataset(name=Data_name+'_96',data_root=f'./Data/MMD/', period=period)
                dataset = AlternatingDataset(dataset1, dataset2, dataset3)
            else:
                raise ValueError
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                collate_fn=custom_collate_fn)
    else:
        Data_name = data_dict[args.dataset_name]
        if args.dataset_name.split('_')[0] in ['ETTh1', 'ETTm1','traffic', 'airquality', 'exchangerate', 'weather', 'electricity', 'nationalillness', 'weather']:
            dataset = T2SDataset(name=Data_name, data_root=f'./Data/our/', period=period)
        else:
            if args.dataset_name == 'SUSHI':
                dataset = T2SDataset(name=Data_name, data_root=f'./Data/SUSHI/', period=period)
            elif args.dataset_name.split('-')[0] == 'TSL':
                dataset = T2SDataset(name=Data_name, data_root=f'./Data/TSL/', period=period)
            elif args.dataset_name.split('-')[0] == 'MMD':
                dataset = T2SDataset(name=Data_name, data_root=f'./Data/MMD/', period=period)
            else:
                raise ValueError
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return dataset, dataloader

def custom_collate_fn(batch):
    grouped_data = {0: [], 1: [], 2: []}
    grouped_data = {
        idx: [data for data, dataset_idx in batch if dataset_idx == idx]
        for idx in grouped_data.keys()
    }
    batches = []
    for idx, data_list in grouped_data.items():
        if data_list:
            batch_texts, batch_xs, batch_embeddings = zip(*data_list)
            batch_texts = [torch.from_numpy(text) if isinstance(text, np.ndarray) else text for text in batch_texts]
            batch_xs = torch.stack(
                [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in batch_xs]
            )
            batch_embeddings = torch.stack(
                [torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding for embedding in batch_embeddings]
            )
            batches.append((batch_texts, batch_xs, batch_embeddings))
    return batches


if __name__ == "__main__":
    pass