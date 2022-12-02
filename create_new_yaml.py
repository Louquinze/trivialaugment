import yaml
from yaml.loader import SafeLoader
import os

if __name__ == '__main__':
    for file in os.listdir("confs"):
        # print(file)
        # Open the file and load the file
        with open(f"confs/{file}") as f:
            data = yaml.load(f, Loader=SafeLoader)
            print(data)
        # Todo set gpu = 2 for 128 option
        with open(f"confs/{file.replace('128', '64')}", 'w') as f:
            data["batch"] = 64
            data = yaml.dump(data, f, sort_keys=False, default_flow_style=False)