import lightning as L
import numpy as np
import gc
import time
import torch
# import torch.nn.functional as F
from minerva.data.datasets.supervised_dataset import SupervisedReconstructionDataset
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.models.nets.image.segment_anything.sam_lora import SAMLoRA
from minerva.transforms.transform import _Transform
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from torch.utils.data import DataLoader
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import os
import random
from scipy.ndimage.interpolation import zoom
import cv2
from patchify import patchify
# from lightning.pytorch.loggers import TensorBoardLogger
# from matplotlib.colors import ListedColormap
from tqdm import tqdm
import argparse
import json

print("Xama! Pode executar agora teus codigo!")