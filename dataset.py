import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __len__(self):
    return self.encodings['input_ids'].shape[0]
    
  def __getitem__(self, i):
    return { key: tensor[i] for key, tensor in self.encodings.items()}

