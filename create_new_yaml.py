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
        if "128" in file:
            with open(f"confs/gpu2_{file}", 'w') as f:
                data["gpu"] = 2
                data = yaml.dump(data, f, sort_keys=False, default_flow_style=False)