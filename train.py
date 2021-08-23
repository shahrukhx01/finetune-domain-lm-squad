import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from transformers import BertForQuestionAnswering


def train_lm_squad(
    train_dataset,
    tokenizer,
    epochs=3,
    pretrained_model_name="shahrukhx01/chemical-bert-uncased",
    save_model_name="chemical-bert-uncased-squad",
):

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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
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

    model.save_pretrained(save_model_name)
    tokenizer.save_pretrained(save_model_name)

    return model, device
