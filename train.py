import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from transformers import BertForQuestionAnswering
import numpy as np


def train_lm_squad(
    train_dataset,
    tokenizer,
    epochs=10,
    pretrained_model_name="shahrukhx01/chemical-bert-uncased",
    save_model_name="chemical-bert-uncased-squad",
):
    def train_collate(batch):
        len_batch = len(batch)
        batch = list(filter(lambda x: x is not None, batch))

        if len_batch > len(batch):
            db_len = len(train_dataset)
            diff = len_batch - len(batch)
            while diff != 0:
                a = train_dataset[np.random.randint(0, db_len)]
                if a is None:
                    continue
                batch.append(a)
                diff -= 1

        return torch.utils.data.dataloader.default_collate(batch)

    model = BertForQuestionAnswering.from_pretrained(pretrained_model_name)

    # setup GPU/CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optim = AdamW(model.parameters(), lr=5e-5)

    # initialize data loader for training data
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=train_collate
    )

    for epoch in range(epochs):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            try:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                # pull all the tensor batches required for training
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                # train model on batch and return outputs (incl. loss)
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                # extract loss
                loss = outputs[0]
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                # print relevant info to progress bar
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
            except:
                continue

        model.save_pretrained(f"{save_model_name}_{epoch}")
        tokenizer.save_pretrained(f"{save_model_name}_{epoch}")

    return model, device
