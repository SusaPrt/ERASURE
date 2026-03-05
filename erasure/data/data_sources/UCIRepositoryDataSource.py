from .datasource import DataSource
from erasure.utils.config.global_ctx import Global
from erasure.data.datasets.Dataset import DatasetWrapper 
from erasure.utils.config.local_ctx import Local
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import ConcatDataset, TensorDataset
import torch
import pandas as pd
import numpy as np
from datasets import Dataset

class UCIWrapper(DatasetWrapper):
    def __init__(self, data, preprocess,label, data_columns):
        super().__init__(data,preprocess)
        self.label = label
        self.data_columns = data_columns

    def __realgetitem__(self, index: int):
        sample = self.data[index]

        X = torch.Tensor([value for key,value in sample.items() if key in self.data_columns])

        y = sample[self.label]
        
        return X,y


class UCIRepositoryDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.id = self.local_config['parameters']['id']
        self.dataset = None
        self.label = self.local_config['parameters']['label']
        self.data_columns = self.local_config['parameters']['data_columns']
        self.to_encode = self.local_config['parameters']['to_encode']

    def get_name(self):
        return self.name

    def create_data(self):

        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)

        pddataset = pd.DataFrame(self.dataset.data.original)

        if not self.data_columns:
            self.data_columns = [col for col in pddataset if col != self.label]
        else:
            self.data_columns = [col for col in pddataset if col in self.data_columns and col != self.label]
            
        
        self.name = self.dataset.metadata.name if 'name' in self.dataset.metadata else 'Name not found'


        hfdataset = Dataset.from_pandas(pddataset)
        
        self.dataset = ConcatDataset( [ hfdataset ] )

        self.dataset.classes = pddataset[self.label].unique()

        return self.get_simple_wrapper(self.dataset)

    
    def get_simple_wrapper(self, data):
        return UCIWrapper(data, self.preprocess, self.label, self.data_columns)

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['label'] = self.local_config['parameters'].get('label','')
        self.local_config['parameters']['data_columns'] = self.local_config['parameters'].get('data_columns',[])
        self.local_config['parameters']['to_encode'] = self.local_config['parameters'].get('to_encode',[])

##Adult has a lot of errors in its data, therefore it's best to handle them in a different loader.
class UCI_Adult_DataSource(UCIRepositoryDataSource):
    
    # column transformer 
    
    def create_data(self):

        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)

        pddataset = pd.DataFrame(self.dataset.data.original)

        pddataset['native-country'] = pddataset['native-country'].apply(lambda x: 'United-States' if x == 'United-States' else 'Other')
        pddataset = pd.get_dummies(pddataset, columns=self.to_encode)

        if not self.data_columns:
            self.data_columns = [col for col in pddataset if col != self.label]

        # normalize the numerical columns 
        for col in self.data_columns:
            if pddataset[col].dtype == 'float64' or pddataset[col].dtype == 'int64':
                pddataset[col] = (pddataset[col] - pddataset[col].mean()) / pddataset[col].std()
            
        pddataset[self.label] = pddataset[self.label].apply(lambda x: 0 if '<' in x else 1)
        
        self.name = self.dataset.metadata.name if 'name' in self.dataset.metadata else 'Name not found'


        hfdataset = Dataset.from_pandas(pddataset)
        
        self.dataset = ConcatDataset( [ hfdataset ] )
        
        self.dataset.classes = pddataset[self.label].unique()

        return self.get_simple_wrapper(self.dataset)
    
"""
id: 755 url: https://archive.ics.uci.edu/dataset/755/accelerometer+gyro+mobile+phone+dataset
"""
class UCI_HAR_DataSource(UCIRepositoryDataSource):
    def create_data(self):
        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)
            self.name = self.dataset.metadata.get('name')

        pddataset = pd.DataFrame(self.dataset.data.original)

        # Metadata keys: dict_keys(['uci_id', 'name', 'repository_url', 'data_url', 'abstract', 'area', 'tasks', 
        # 'characteristics', 'num_instances', 'num_features', 'feature_types', 'demographics', 'target_col', 
        # 'index_col', 'has_missing_values', 'missing_values_symbol', 'year_of_dataset_creation', 'last_updated', 
        # 'dataset_doi', 'creators', 'intro_paper', 'additional_info'])

        print("[DEBUG] Metadata uci_id:", self.dataset.metadata.get('uci_id'))
        print("[DEBUG] Metadata name:", self.dataset.metadata.get('name'))
        print("[DEBUG] Metadata repo url:", self.dataset.metadata.get('repository_url'))
        print("[DEBUG] Metadata characteristics:", self.dataset.metadata.get('characteristics'))
        print("[DEBUG] Metadata tasks:", self.dataset.metadata.get('tasks'))

        print("[DEBUG] UCI_HAR_DataSource: columns:", list(pddataset.columns))
        print("[DEBUG] UCI_HAR_DataSource: shape:", pddataset.shape)
        print("[DEBUG] UCI_HAR_DataSource: Classes in Dataset: {}".format(pddataset[self.label].unique()))

        #print("[DEBUG] dtypes:\n", pddataset.dtypes)
        #print("[DEBUG] Missing values:\n", pddataset.isnull().sum())
        
        #print("[DEBUG] UCI_HAR_DataSource: head:\n", pddataset.head())
        #print("[DEBUG] UCI_HAR_DataSource: tail:\n", pddataset.tail())

        window_size = self.local_config['parameters'].get('window_size', 128)
        
        windows = []
        y = []
        data_array = pddataset[self.data_columns].to_numpy()
        label_array = pddataset[self.label].to_numpy().astype(int)
        for start in range(0, len(data_array) - window_size + 1, window_size):
            end = start + window_size
            windows.append(data_array[start:end])
            window_labels = label_array[start:end]
            majority_label = np.bincount(window_labels).argmax()
            y.append(majority_label)

        print("[DEBUG] UCI_HAR_DataSource: after windowing, dataset shape:", np.stack(windows).shape)

        X = np.stack(windows)           # (samples, window_size, n_features)
        X = X.transpose(0,2,1)          # (samples, n_features, window_size)
        y = np.array(y)
        
        print("[DEBUG] UCI_HAR_DataSource: final data shape:", X.shape, y.shape)

        X = torch.Tensor(X).long()
        y = torch.Tensor(y).long()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        y = y.squeeze()
        
        self.dataset = TensorDataset(X,y)

        self.dataset.classes = pddataset[self.label].unique()
    
        print("[DEBUG] UCI_HAR_DataSource: final dataset classes:", self.dataset.classes)
        print("[DEBUG] UCI_HAR_DataSource: final dataset length:", len(self.dataset))

        return self.get_simple_wrapper(self.dataset)
    
    def get_simple_wrapper(self, data):
        x = data[0][0]
        if x.ndim == 2:
            return DatasetWrapper(data, self.preprocess)

        elif x.ndim == 1:
            return UCIWrapper(data, self.preprocess, self.label, self.data_columns)