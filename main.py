from load_encode_data import read_squad, add_end_idx, add_token_positions
from transformers import BertTokenizerFast
from data_set import SquadDataset
from train import train_lm_squad
from eval import evaluate_model

if __name__ == "__main__":
    model_name = "shahrukhx01/chemical-bert-uncased"

    ## load data
    train_contexts, train_questions, train_answers = read_squad(
        "data/squad/train-v2.0.json"
    )
    val_contexts, val_questions, val_answers = read_squad("data/squad/dev-v2.0.json")

    ## add end position for answers in SQUAD data
    train_answers, train_contexts = add_end_idx(train_answers, train_contexts)
    val_answers, val_contexts = add_end_idx(val_answers, val_contexts)

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    ## encode text data
    train_encodings = tokenizer(
        train_contexts, train_questions, truncation=True, padding=True, max_length=512
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

    ## train model
    model, device = train_lm_squad(
        train_dataset,
        tokenizer,
        pretrained_model_name=model_name,
        save_model_name="chemical-bert-uncased-squad",
    )
    ## evaluate model on validation set
    start_pred, end_pred, start_true, end_true = evaluate_model(
        model, val_dataset, device
    )

    print("T/F\tstart\tend\n")
    for i in range(len(start_true)):
        print(
            f"true\t{start_true[i]}\t{end_true[i]}\n"
            f"pred\t{start_pred[i]}\t{end_pred[i]}\n"
        )
