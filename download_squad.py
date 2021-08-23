import os
import requests
import json

if not os.path.exists("../data/squad"):
    os.mkdir("data/squad/")

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
res = requests.get(f"{url}train-v2.0.json")

for file in ["train-v2.0.json", "dev-v2.0.json"]:
    res = requests.get(f"{url}{file}")
    # write to file
    file_path = f"data/squad/{file}"
    with open(file_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=4):
            f.write(chunk)
