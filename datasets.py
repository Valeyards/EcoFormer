import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class RNADataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=True):
        self.data_dir = data_dir
        self.transform = transform
        
        self.samples_df = pd.read_csv(csv_file)
        
        self.labels = self.samples_df['label'].values
        self.sample_ids = self.samples_df['sample.inf.new.id'].values
        
        self.rna_types = {
            'mrna': os.path.join(data_dir, 'mrna'),
            'lncrna': os.path.join(data_dir, 'lncrna'),
            'circrna': os.path.join(data_dir, 'circrna')
        }
        
        if transform:
            self.stats = {}
            for rna_type, path in self.rna_types.items():
                all_data = []
                for sample_id in self.sample_ids:
                    data_path = os.path.join(path, f"{sample_id}.npy")
                    data = np.load(data_path)
                    all_data.append(data)
                all_data = np.vstack(all_data)
                
                non_zero_data = all_data[all_data > 0]
                if len(non_zero_data) > 0:
                    self.stats[rna_type] = {
                        'log_mean': np.mean(np.log1p(non_zero_data)),
                        'log_std': np.std(np.log1p(non_zero_data))
                    }
                else:
                    self.stats[rna_type] = {
                        'log_mean': 0,
                        'log_std': 1
                    }
    
    def __len__(self):
        return len(self.sample_ids)

    def transform_data(self, data, rna_type):
        if not self.transform:
            return data
        
        data_log = np.log1p(data)
        data_normalized = (data_log - self.stats[rna_type]['log_mean']) / (self.stats[rna_type]['log_std'] + 1e-8)
        
        return data_normalized
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        label = self.labels[idx]
        
        rna_data = {}
        for rna_type, path in self.rna_types.items():
            data_path = os.path.join(path, f"{sample_id}.npy")
            data = np.load(data_path)
            if self.transform:
                data = self.transform_data(data, rna_type)
            rna_data[rna_type] = data
        
        data_tensors = {rna_type: torch.tensor(data, dtype=torch.float32) 
                       for rna_type, data in rna_data.items()}
        
        return {
            'sample_id': sample_id,
            'data': data_tensors,
            'label': torch.tensor(label, dtype=torch.long),
            'cancer_type': self.samples_df.iloc[idx]['cohort.new']
        }
    
def create_external_data_loader(csv_file, data_dir, batch_size, num_workers):
    dataset = RNADataset(
        csv_file=csv_file,
        data_dir=data_dir
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

def create_data_loaders(root_dir, data_dir, batch_size=32, num_workers=4, eval=False):
    loaders = {}
    for split in ['train', 'val']:
            csv_file = os.path.join(root_dir, f'{split}.csv')
            dataset = RNADataset(csv_file=csv_file, data_dir=data_dir)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train' and not eval), #False
                num_workers=num_workers,
                pin_memory=True
            )
            
            loaders[split] = loader
    
    return loaders


if __name__ == '__main__':
    root_dir = '../data/splits' 
    data_dir = '../data/expression_npy' 
    
    data_loaders = create_data_loaders(
        root_dir=root_dir,
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    batch = next(iter(train_loader))