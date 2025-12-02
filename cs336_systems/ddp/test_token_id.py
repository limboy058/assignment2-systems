import os
import numpy as np

data_dir = "/mnt/mnt/zjdx"

train_path = os.path.join(data_dir, "owt_train.bin")
valid_path = os.path.join(data_dir, "owt_valid.bin")

train_data = np.memmap(train_path, dtype="uint16", mode="r")
valid_data = np.memmap(valid_path, dtype="uint16", mode="r")

print("train max token:", train_data.max())
print("valid max token:", valid_data.max())
