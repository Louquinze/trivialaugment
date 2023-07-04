import os

import pandas as pd
import torch

path = "save"
res = None

for file in os.listdir(path):
    if "e200" not in file:
        continue
    data = torch.load(path + "/" + file, map_location="cpu")
    # func, network
    d = {
        "network": [file.split("_")[2]],
        "func": [file.split("_")[1]],
        "seed": [file.split("_")[0]],
        "epoch": [data["epoch"]],
        "test_top1": [data["log"]["test"]["top1"]],
        "eval_test_top1": [data["log"]["test"]["eval_top1"]],
        "file": [file]
    }

    d = pd.DataFrame(d)

    if res is None:
        res = d
    else:
        res = pd.concat((res, d), ignore_index=True)

res.to_csv(path + ".csv")