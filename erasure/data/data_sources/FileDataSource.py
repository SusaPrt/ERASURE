from pathlib import Path
import numpy as np
from .datasource import DataSource
from erasure.data.datasets.Dataset import DatasetExtendedWrapper, DatasetWrapper 
from torch.utils.data import ConcatDataset, TensorDataset
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
import inspect 
import torch
from torchvision.transforms import Compose
import pandas as pd

class CSVDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.path = self.local_config['parameters']['path']
        self.data_columns = self.local_config['parameters']['data_columns']
        self.label_columns  = self.local_config['parameters']['label_columns']

    def get_name(self):
        return self.path.split(".")[-1] 

    def create_data(self):
        self.data = pd.read_csv(self.path, index_col = 0)
        self.data_columns = [col for col in self.data.columns if col != self.label_column] if not self.data_columns else self.data_columns
        self.label_columns = [self.data.columns[-1]] if not self.label_columns else self.label_columns

        dataset = CSVDatasetWrapper(self.data, self.label_columns, self.data_columns, self.preprocess)
        return dataset
    

    def get_simple_wrapper(self, data):
        data_csv = self.data.loc[data.indices]
        return CSVDatasetWrapper(data_csv, self.label_columns, self.data_columns, self.preprocess)
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['root_path'] = self.local_config.get('root_path','resources/data')
        self.local_config['parameters']['label_columns'] = self.local_config['parameters'].get('label_columns', 'targets')
        self.local_config['parameters']['data_columns'] = self.local_config['parameters'].get('data_columns', [])
    
  
class CSVDatasetWrapper(DatasetWrapper):
    def __init__(self, data, label_columns, data_columns, preprocess = []):
        self.data = data 
        self.preprocess = preprocess
        self.data_columns = data_columns
        self.label_columns = label_columns
        self.classes =  self.data[self.label_columns[0]].unique() 

    def __realgetitem__(self, index: int):
        row = self.data.iloc[index]  
        x = row[self.data_columns].values  
        y = row[self.label_columns].values
        x = x[0]

        return x, y

    def get_n_classes(self):
        return len(self.classes)

class HAR_CSV_DataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.path = self.local_config['parameters']['path']
        self.id_columns = self.local_config['parameters'].get('id_columns', [])
        self.label_columns = self.local_config['parameters']['label_columns']
        self.pos_columns = self.local_config['parameters'].get('pos_columns', [])
        self.data_columns = self.local_config['parameters']['data_columns']
        self.window_size = self.local_config['parameters']['window_size']

        print("[DEBUG] Initializing HAR_CSV_DataSource with parameters:")
        print("[DEBUG] Data columns:", self.data_columns)
        print("[DEBUG] Label columns:", self.label_columns)
        print("[DEBUG] ID columns:", self.id_columns)
        print("[DEBUG] Position columns:", self.pos_columns)
        print("[DEBUG] Window size:", self.window_size)

    def create_data(self):
        self.data = pd.read_csv(self.path, index_col = False, header=0)
        print("[DEBUG] HAR_CSV_DataSource: Original data shape:", self.data.shape)
        print("[DEBUG] HAR_CSV_DataSource: Data columns available:", self.data.columns.tolist())

        if self.pos_columns and not self.data[self.pos_columns].dtype.kind in 'biufc':
            unique_positions = pd.Series(self.data[self.pos_columns].values.ravel()).unique()
            position_mapping = {pos: idx for idx, pos in enumerate(unique_positions)}
            print("[DEBUG] HAR_CSV_DataSource: Position mapping:", position_mapping)
            if isinstance(self.pos_columns, list):
                for col in self.pos_columns:
                    self.data[col] = self.data[col].map(position_mapping)
            else:
                self.data[self.pos_columns] = self.data[self.pos_columns].map(position_mapping)
            print("[DEBUG] HAR_CSV_DataSource: Unique positions after mapping:", np.unique(self.data[self.pos_columns].values.ravel()))
        
        windows = []
        labels = []
        ids = []
        positions = []
        for start in range(0, len(self.data) - self.window_size + 1, self.window_size):
            end = start + self.window_size
            windows.append(self.data.iloc[start:end][self.data_columns].values)
            window_labels = self.data.iloc[start:end][self.label_columns].values.ravel().astype(int)
            majority_label = np.bincount(window_labels).argmax()
            labels.append(majority_label)
            if self.id_columns:
                window_ids = self.data.iloc[start:end][self.id_columns].values.ravel().astype(int)
                majority_id = np.bincount(window_ids).argmax()
                ids.append(majority_id)
            if self.pos_columns:
                window_positions = self.data.iloc[start:end][self.pos_columns].values.ravel().astype(int)
                majority_position = np.bincount(window_positions).argmax()
                positions.append(majority_position)

        print("[DEBUG] HAR_CSV_DataSource: Data shape after windowing:", np.stack(windows).shape)
        X = np.stack(windows)           # (samples, window_size, n_features)
        X = X.transpose(0,2,1)          # (samples, n_features, window_size)
        labels = np.array(labels)
        if labels.min() != 0:
            labels = labels - labels.min()
        
        ids = np.array(ids) if self.id_columns else None
        position = np.array(positions) if self.pos_columns else None

        print("[DEBUG] HAR_CSV_DataSource: Final data shape:", X.shape, labels.shape, ids.shape if ids is not None else None, position.shape if position is not None else None)

        X = torch.Tensor(X).long()
        if position is None and ids is not None:
            y_comb = np.stack([labels, ids], axis=0)
            y_comb = y_comb.T
        elif position is not None and ids is None:
            y_comb = np.stack([labels, position], axis=0)
            y_comb = y_comb.T
        else:
            y_comb = labels

        y = torch.Tensor(y_comb).long()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        self.dataset = TensorDataset(X, y)
        self.dataset.data_columns = self.data_columns
        self.dataset.name = self.get_name()
        self.dataset.preprocess = []
        self.dataset.data = X

        classes = np.unique(labels)
        self.dataset.classes = classes

        print("[DEBUG] HAR_CSV_DataSource: Dataset name:", self.dataset.name)
        print("[DEBUG] HAR_CSV_DataSource: Dataset shape:", self.dataset.tensors[0].shape, self.dataset.tensors[1].shape)
        print("[DEBUG] HAR_CSV_DataSource: Classes:", self.dataset.classes)
        print("[DEBUG] HAR_CSV_DataSource: Unique ids:", np.unique(ids) if ids is not None else "N/A")
        print("[DEBUG] HAR_CSV_DataSource: Unique positions:", np.unique(positions) if positions is not None else "N/A")

        dataset = self.get_wrapper(self.dataset)

        return dataset

    def get_simple_wrapper(self, data):
        return DatasetWrapper(data, self.preprocess)
    
    def get_extended_wrapper(self, data):
        return DatasetExtendedWrapper(self.get_simple_wrapper(data))

    def get_name(self):
        return Path(self.path).stem
    