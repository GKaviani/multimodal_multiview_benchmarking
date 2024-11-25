import os

import pandas as pd
import torch

data_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/IMU/IMU_LeftArm"

for activty in os.listdir(data_dir):
    for path , dir , files in os.walk(os.path.join(data_dir , activty)):
        for file in files:
            if file.endswith(".csv"):
                print(file)

