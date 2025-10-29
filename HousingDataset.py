from torch.utils.data import Dataset
import torch

class HousingDataset(Dataset):
    def __init__(self,features, targets):
        super(HousingDataset, self).__init__()
        self.features=torch.from_numpy(features)
        self.targets=torch.from_numpy(targets)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx],self.targets[idx]