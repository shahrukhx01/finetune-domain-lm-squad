from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np


def evaluate_model(model, val_dataset, device):
    def val_collate(batch):
        len_batch = len(batch)
        batch = list(filter(lambda x: x is not None, batch))

        if len_batch > len(batch):
            db_len = len(val_dataset)
            diff = len_batch - len(batch)
            while diff != 0:
                a = val_dataset[np.random.randint(0, db_len)]
                if a is None:
                    continue
                batch.append(a)
                diff -= 1

        return torch.utils.data.dataloader.default_collate(batch)

    # switch model out of training mode
    model.eval()

    # val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=val_collate)

    acc = []

    # initialize loop for progress bar
    loop = tqdm(val_loader)
    # loop through batches
    for batch in loop:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_true = batch["start_positions"].to(device)
            end_true = batch["end_positions"].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull preds out
            start_pred = torch.argmax(outputs["start_logits"], dim=1)
            end_pred = torch.argmax(outputs["end_logits"], dim=1)
            # calculate accuracy for both and append to accuracy list
            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())
    # calculate average accuracy in total
    acc = sum(acc) / len(acc)
    print(f"Overall acc. {acc}")
    return start_pred, end_pred, start_true, end_true
