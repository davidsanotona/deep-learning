import torch
from torch.utils.data import Dataset


class LoanDataset(Dataset):
    def __init__(self, X, y):
        """
        X: Feature matrix (NumPy array or Pandas DataFrame)
        y: Labels (NumPy array or Pandas Series)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]