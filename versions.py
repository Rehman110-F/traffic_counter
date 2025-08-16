# setup.py
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision
import ultralytics
from IPython import display
import ipywidgets as widgets
import tqdm
import scipy
import matplotlib
from deep_sort_realtime.deepsort_tracker import DeepSort
def print_versions():
    print("Python:", sys.version)
    print("OS module: built-in")
    print("Pathlib:", Path)
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pillow:", Image.__version__)
    print("Torch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("IPython:", display.__name__)
    print("ipywidgets:", widgets.__version__)
    print("tqdm:", tqdm.__version__)
    print("SciPy:", scipy.__version__)
    print("Matplotlib:", matplotlib.__version__)

def device_setup():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype

if __name__ == "__main__":
    print_versions()
    device_setup()


