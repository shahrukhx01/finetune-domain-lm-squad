from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


def evaluate_model(model, val_dataset, device):
    # switch model out of training mode
    model.eval()

    # val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=16)

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
