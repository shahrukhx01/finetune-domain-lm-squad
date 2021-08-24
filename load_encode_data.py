import json
from tqdm import tqdm


def read_squad(path):
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in tqdm(squad_dict["data"]):
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                if "plausible_answers" in qa.keys():
                    access = "plausible_answers"
                else:
                    access = "answers"
                for answer in qa["answers"]:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers


def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer["text"]
        # we already know the start index
        start_idx = answer["answer_start"]
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer["answer_end"] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx - n : end_idx - n] == gold_text:
                    # this means the answer is off by 'n' tokens
                    answer["answer_start"] = start_idx - n
                    answer["answer_end"] = end_idx - n

        return answers, contexts


def add_token_positions(encodings, answers, tokenizer):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        print(answers)
        start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"]))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift one token forward
        go_back = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(
                i, answers[i]["answer_end"] - go_back
            )
            go_back += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )

    return encodings, answers
