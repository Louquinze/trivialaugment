import os
import pandas as pd
import json

if __name__ == '__main__':
    os.chdir("logs")
    df = {"func": [], "dataset": [], "seed": [], "top1-test": [], "network": []}
    for item in os.listdir():
        item_split = item.split("_")
        if 'svhn' in item:
            dataset = 'cifar10'
        elif "cifar100" in item:
            dataset = 'cifar100'
        else:
            dataset = "cifar10"


        seed = int(item_split[-1])
        if "Func" in item_split:
            func = item_split[-3:-1]
            func = "_".join(func)
        else:
            func = item_split[-2]
        os.chdir(item)
        try:
            with open("res.json") as f:
                data = json.load(f)
                df["func"].append(func)
                df["dataset"].append(dataset)
                df["seed"].append(seed)
                df["top1-test"].append(data["top1_test"])
                df["network"].append(item_split[0])
        except:
            print(os.getcwd())
        os.chdir('..')

    df = pd.DataFrame(df)
    df["num_seeds"] = 1
    print(df.sort_values(["top1-test"], ascending=False).groupby(['dataset', 'func'],
                                                                 as_index=False).agg({"top1-test": ['mean', 'std'],
                                                                                      "num_seeds": ['sum']},
                                                                                     as_index=False))
    # print(df.groupby(['dataset', 'func'], as_index=False).agg({"top1-test": ['mean', 'std']}).sort_values(["dataset", "top1-test"], ascending=False).to_latex())
    df.sort_values(["top1-test"], ascending=False).groupby(['dataset', 'func'],
                                                           as_index=False).agg({"top1-test": ['mean', 'std'],
                                                                                "num_seeds": ['sum']},
                                                                               as_index=False).to_csv(
        "../res_logs.tsv", sep="\t")
