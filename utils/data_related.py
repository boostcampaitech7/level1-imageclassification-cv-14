from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

def get_dataloader(dataset : Dataset, **kwargs):
    return DataLoader(dataset, **kwargs)

def data_split(dataset, test_size, stratify, **kwargs):

    return train_test_split(dataset, test_size=test_size, stratify=stratify, **kwargs)

def get_subset(dataset, idx):
    return Subset(dataset, idx)

