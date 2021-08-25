from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from load_encode_data import read_squad, add_end_idx, add_token_positions
from transformers import BertTokenizerFast
from data_set import SquadDataset
from transformers import BertForQuestionAnswering


def evaluate_model(model, val_dataset, device):
    model.to(device)

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


if __name__ == "__main__":
    for i in range(9):
        model_name = f"./chemical-bert-uncased-squad_{i}"

        ## load data
        train_contexts, train_questions, train_answers = read_squad(
            "data/squad/train-v2.0.json"
        )
        val_contexts, val_questions, val_answers = read_squad(
            "data/squad/dev-v2.0.json"
        )

        ## add end position for answers in SQUAD data
        train_answers, train_contexts = add_end_idx(train_answers, train_contexts)
        val_answers, val_contexts = add_end_idx(val_answers, val_contexts)

        tokenizer = BertTokenizerFast.from_pretrained(model_name)

        ## encode text data
        train_encodings = tokenizer(
            train_contexts,
            train_questions,
            truncation=True,
            padding=True,
            max_length=512,
        )
        val_encodings = tokenizer(
            val_contexts, val_questions, truncation=True, padding=True, max_length=512
        )

        # add token positions to encodings
        train_encodings, train_answers = add_token_positions(
            train_encodings, train_answers, tokenizer
        )
        val_encodings, val_answers = add_token_positions(
            val_encodings, val_answers, tokenizer
        )

        ## add torch dataset wrapper around train/val encodings
        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)

        # setup GPU/CPU
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = BertForQuestionAnswering.from_pretrained(model_name)

        ## evaluate model on validation set
        start_pred, end_pred, start_true, end_true = evaluate_model(
            model, val_dataset, device
        )

        """print("T/F\tstart\tend\n")
        for i in range(len(start_true)):
            print(
                f"true\t{start_true[i]}\t{end_true[i]}\n"
                f"pred\t{start_pred[i]}\t{end_pred[i]}\n"
            )"""
