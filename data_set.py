import torch


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        try:
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        except:
            pass

    def __len__(self):
        return len(self.encodings.input_ids)
