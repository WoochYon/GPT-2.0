import torch
from torch.utils.data import Dataset, DataLoader

class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

def get_data_loaders(file_path, block_size, batch_size=64):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_eco = f.read()

        print(f"File length: {len(text_eco)} characters")

        chars_eco = sorted(list(set(text_eco)))
        vocab_size_eco = len(chars_eco)
        print(f"Unique characters: {vocab_size_eco}")

        stoi_eco = {ch: i for i, ch in enumerate(chars_eco)}
        itos_eco = {i: ch for ch, i in stoi_eco.items()}

        data_eco = torch.tensor([stoi_eco[ch] for ch in text_eco], dtype=torch.long)
        print(f"Data tensor shape: {data_eco.shape}")

        dataset = NextTokenDataset(data_eco, block_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return loader, vocab_size_eco, stoi_eco, itos_eco

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}.")
        return None, None, None, None
