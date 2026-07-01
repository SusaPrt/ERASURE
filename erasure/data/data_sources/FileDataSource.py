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
        self.labels  = self.local_config['parameters']['labels']

    def get_name(self):
        return self.path.split(".")[-1] 

    def create_data(self):
        self.data = pd.read_csv(self.path, index_col = 0)
        self.data_columns = [col for col in self.data.columns if col != self.label_column] if not self.data_columns else self.data_columns
        self.labels = [self.data.columns[-1]] if not self.labels else self.labels

        dataset = CSVDatasetWrapper(self.data, self.labels, self.data_columns, self.preprocess)
        return dataset
    

    def get_simple_wrapper(self, data):
        data_csv = self.data.loc[data.indices]
        return CSVDatasetWrapper(data_csv, self.labels, self.data_columns, self.preprocess)
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['root_path'] = self.local_config.get('root_path','resources/data')
        self.local_config['parameters']['labels'] = self.local_config['parameters'].get('labels', 'targets')
        self.local_config['parameters']['data_columns'] = self.local_config['parameters'].get('data_columns', [])
    
  
class CSVDatasetWrapper(DatasetWrapper):
    def __init__(self, data, labels, data_columns, preprocess = []):
        self.data = data 
        self.preprocess = preprocess
        self.data_columns = data_columns
        self.labels = labels
        self.classes =  self.data[self.labels[0]].unique() 

    def __realgetitem__(self, index: int):
        row = self.data.iloc[index]  
        x = row[self.data_columns].values  
        y = row[self.labels].values
        x = x[0]

        return x, y

    def get_n_classes(self):
        return len(self.classes)

class HAR_CSV_DataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.path = self.local_config['parameters']['path']
        self.id = self.local_config['parameters'].get('id', [])
        self.labels = self.local_config['parameters']['label']
        self.pos = self.local_config['parameters'].get('pos', [])
        self.data_columns = self.local_config['parameters']['data']
        self.window_size = self.local_config['parameters']['window_size']

        self.min_class_ratio = self.local_config['parameters'].get('min_class_ratio', 0.25)
        self.drop_underrepresented_classes = self.local_config['parameters'].get('drop_underrepresented_classes', True)

        print("[DEBUG] Initializing HAR_CSV_DataSource with parameters:")
        print("[DEBUG] Data columns:", self.data_columns)
        print("[DEBUG] Label columns:", self.labels)
        print("[DEBUG] ID columns:", self.id)
        print("[DEBUG] Position columns:", self.pos)
        print("[DEBUG] Window size:", self.window_size)
        print("[DEBUG] Drop underrepresented classes:", self.drop_underrepresented_classes)
        print("[DEBUG] Min class ratio:", self.min_class_ratio)

    def _filter_underrepresented_classes(self, labels, windows, ids=None, positions=None):
        if not self.drop_underrepresented_classes:
            return labels, windows, ids, positions

        labels_array = np.array(labels)
        if labels_array.size == 0:
            return labels, windows, ids, positions

        class_counts = np.bincount(labels_array)
        if class_counts.size == 0:
            return labels, windows, ids, positions

        max_count = class_counts.max()
        threshold = max_count * self.min_class_ratio
        underrepresented_labels = [
            int(label)
            for label, count in enumerate(class_counts)
            if count > 0 and count < threshold
        ]

        if not underrepresented_labels:
            return labels, windows, ids, positions

        keep_mask = np.isin(labels_array, [label for label in np.unique(labels_array) if label not in underrepresented_labels])
        filtered_labels = [label for label, keep in zip(labels, keep_mask) if keep]
        filtered_windows = [window for window, keep in zip(windows, keep_mask) if keep]

        filtered_ids = None
        if ids is not None:
            filtered_ids = [identifier for identifier, keep in zip(ids, keep_mask) if keep]

        filtered_positions = None
        if positions is not None:
            filtered_positions = [position for position, keep in zip(positions, keep_mask) if keep]

        label_mapping = {
            int(old_label): int(new_label)
            for new_label, old_label in enumerate(sorted(set(labels_array.tolist()) - set(underrepresented_labels)))
        }
        filtered_labels = [label_mapping[label] for label in filtered_labels]

        print("[DEBUG] HAR_CSV_DataSource: Dropped underrepresented labels:", underrepresented_labels)
        print("[DEBUG] HAR_CSV_DataSource: Label remapping:", label_mapping)
        print("[DEBUG] HAR_CSV_DataSource: Remaining class counts:", {int(label): int(np.sum(np.array(filtered_labels) == label)) for label in np.unique(filtered_labels)})

        return filtered_labels, filtered_windows, filtered_ids, filtered_positions

    def create_data(self):
        self.data = pd.read_csv(self.path, index_col = False, header=0)
        print("[DEBUG] HAR_CSV_DataSource: Original data shape:", self.data.shape)
        print("[DEBUG] HAR_CSV_DataSource: Data columns available:", self.data.columns.tolist())

        if self.pos and not self.data[self.pos].dtype.kind in 'biufc':
            unique_positions = pd.Series(self.data[self.pos].values.ravel()).unique()
            position_mapping = {pos: idx for idx, pos in enumerate(unique_positions)}
            print("[DEBUG] HAR_CSV_DataSource: Position mapping:", position_mapping)
            if isinstance(self.pos, list):
                for col in self.pos:
                    self.data[col] = self.data[col].map(position_mapping)
            else:
                self.data[self.pos] = self.data[self.pos].map(position_mapping)
            print("[DEBUG] HAR_CSV_DataSource: Unique positions after mapping:", np.unique(self.data[self.pos].values.ravel()))
        
        windows = []
        labels = []
        ids = []
        positions = []
        for start in range(0, len(self.data) - self.window_size + 1, self.window_size):
            end = start + self.window_size
            windows.append(self.data.iloc[start:end][self.data_columns].values)
            window_labels = self.data.iloc[start:end][self.labels].values.ravel().astype(int)
            majority_label = np.bincount(window_labels).argmax()
            labels.append(majority_label)
            if self.id:
                window_ids = self.data.iloc[start:end][self.id].values.ravel().astype(int)
                majority_id = np.bincount(window_ids).argmax()
                ids.append(majority_id)
            if self.pos:
                window_positions = self.data.iloc[start:end][self.pos].values.ravel().astype(int)
                majority_position = np.bincount(window_positions).argmax()
                positions.append(majority_position)

        print("[DEBUG] HAR_CSV_DataSource: Data shape after windowing:", np.stack(windows).shape)

        labels, windows, ids, positions = self._filter_underrepresented_classes(labels, windows, ids, positions)

        # balancing
        class_counts = np.bincount(np.array(labels))
        min_class_count = class_counts[class_counts > 0].min()
        balanced_windows = []
        balanced_labels = []
        balanced_ids = []
        balanced_positions = []
        for class_label in np.unique(labels):
            class_indices = np.where(np.array(labels) == class_label)[0]
            if len(class_indices) > min_class_count:
                selected_indices = np.random.choice(class_indices, min_class_count, replace=False)
            else:
                selected_indices = class_indices
            balanced_windows.extend([windows[i] for i in selected_indices])
            balanced_labels.extend([labels[i] for i in selected_indices])
            if self.id:
                balanced_ids.extend([ids[i] for i in selected_indices])
            if self.pos:
                balanced_positions.extend([positions[i] for i in selected_indices])

        print("[DEBUG] HAR_CSV_DataSource: Data shape, after balancing:", np.stack(balanced_windows).shape)

        X = np.stack(balanced_windows)           # (samples, window_size, n_features)
        X = X.transpose(0,2,1)          # (samples, n_features, window_size)
        labels = np.array(balanced_labels)
        
        ids = np.array(balanced_ids) if self.id else None
        position = np.array(balanced_positions) if self.pos else None

        print("[DEBUG] HAR_CSV_DataSource: Final data shape:", X.shape, labels.shape, ids.shape if ids is not None else None, position.shape if position is not None else None)

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