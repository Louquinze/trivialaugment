with open("TrivialAugment/networks/activations.py", "r") as f:
    with open("func.csv", "w") as r:
        Lines = f.readlines()
        is_class = False
        comment_count = 0
        func = None
        is_head = True
        for line in Lines:
            if "class" in line:
                is_class = True
                comment_count = 0
                # print(line.split(" ")[1].replace("(nn.Module):\n", ""))
                func = line.split(" ")[1].replace("(nn.Module):\n", "")
            if '"""' in line and comment_count==1:
                comment_count = 0
                is_class = False
                func = None
            if is_class and comment_count==1 and "arch_lr" not in line:
                row = line.split()
                row[0] = func
                r.write(",".join(row) + "\n")
            if '"""' in line and comment_count==0:
                comment_count += 1
