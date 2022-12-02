import yaml
from yaml.loader import SafeLoader
import os

if __name__ == '__main__':
    for file in os.listdir("confs"):
        if "relu" in file or "swish" in file or "func" in file:
            continue
        # print(file)
        # Open the file and load the file
        with open(f"confs/{file}") as f:
            data = yaml.load(f, Loader=SafeLoader)
            print(data)

        if "128" in file:
            with open(f"confs/swish_{file}", 'w') as f:
                data["ac_func"] = "SiLU"
                data = yaml.dump(data, f, sort_keys=False, default_flow_style=False)