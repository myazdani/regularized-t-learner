import torch
from torch.utils.data import Dataset
from sklift.datasets import fetch_lenta
from sklearn.model_selection import train_test_split
import pandas as pd

class FetchLenta:
    def __init__(self):
        self.dataset = fetch_lenta()
        
        # only deal with numeric cols:
        numeric_cols = self.dataset.data.columns[self.dataset.data.apply(lambda x: 
                                            pd.api.types.is_numeric_dtype(x))]
        self.X = self.dataset.data[numeric_cols].fillna(0).values        
        # make treatment binary
        treat_dict = {
            'test': 1,
            'control': 0
        }
        self.dataset.treatment = self.dataset.treatment.map(treat_dict)
        # stratify by two columns: treatment and target.
        # Intuition: In a binary classification problem definition we stratify 
        # train set by splitting target 0/1 column. In uplift modeling we have two
        # columns instead of one.
        self.stratify_cols = pd.concat([self.dataset.treatment, self.dataset.target], axis=1)
        
    def train_test_split(self, **kwargs):
        
        X_train, X_val, trmnt_train, trmnt_val, y_train, y_val = train_test_split(
            self.X,
            self.dataset.treatment,
            self.dataset.target,
            stratify=self.stratify_cols,
            **kwargs
            )
        
        return X_train, X_val, trmnt_train, trmnt_val, y_train, y_val
    
    
class TreatmentDataset(Dataset):
    def __init__(self, X, y, t):
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        self.t = torch.Tensor(t) # treatment 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx],self.t[idx]
