import torch


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        data = {}
        for key, val in self.encodings.items():
            if (val[idx]) >= 1000000:
                data[key] = 1
            data[key] = torch.tensor(val[idx])

        return data

    def __len__(self):
        return len(self.encodings.input_ids)
