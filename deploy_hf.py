from transformers import BertForQuestionAnswering, BertTokenizerFast


model = BertForQuestionAnswering.from_pretrained("chemical-bert-uncased-squad_0")
model.push_to_hub("chemical-bert-uncased-squad2")

tokenizer = BertTokenizerFast.from_pretrained("./chemical-bert-uncased-squad_0")
tokenizer.push_to_hub("chemical-bert-uncased-squad2")
