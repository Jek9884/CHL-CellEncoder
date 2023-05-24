import random
import torch
from torch.utils.data import Dataset

CLASS = 7
    
class TextSamplerDataset(Dataset):
    '''
        Class to handle dataset examples
    '''
    
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.device = device

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(self.device)
        
        return full_seq

    def __len__(self):
        return self.data.shape[0]