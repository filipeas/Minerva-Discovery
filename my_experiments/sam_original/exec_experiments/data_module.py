import os
import random
from pathlib import Path
import numpy as np
from typing import List, Optional, Tuple
import lightning as L
import torch
from torch.utils.data import DataLoader

from minerva.data.datasets.supervised_dataset import SimpleDataset
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.transforms.transform import _Transform
from minerva.data.readers.reader import _Reader

""" class for apply transpose transform in image """
class Transpose(_Transform):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Verifica se a imagem tem a forma HxWxC
        if len(x.shape) == 3:  # HxWxC
            # Transpõe de HxWxC para CxHxW
            x_transposed = np.transpose(x, (2, 0, 1))
            x_transposed = torch.from_numpy(x_transposed).float()
        else:
            raise ValueError("Input image must have 3 dimensions (HxWxC)")

        return x_transposed

""" class for apply padding transform in image """
class Padding(_Transform):
    def __init__(self, target_h_size: int, target_w_size: int):
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        pad_h = max(0, self.target_h_size - h)
        pad_w = max(0, self.target_w_size - w)
        if len(x.shape) == 2:
            padded = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
            padded = np.expand_dims(padded, axis=2)
            padded = torch.from_numpy(padded).float()
        else:
            padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded = torch.from_numpy(padded).float()

        padded = np.transpose(padded, (2, 0, 1))
        return padded

""" class for create dataset with SAM pattern """
class DatasetForSAM(SimpleDataset):
    def __init__(
            self, 
            readers: List[_Reader], 
            transforms: Optional[_Transform] = None,
            transform_coords_input:Optional[dict]=None,
            multimask_output:bool=True,
    ):
        """
        Custom Dataset to use properties that needed in images when send some image to SAM model.

        Parameters
        ----------
        readers: List[_Reader]
            List of data readers. It must contain exactly 2 readers.
            The first reader for the input data and the second reader for the
            target data.
        transforms: Optional[_Transform]
            Optional data transformation pipeline.
        transform_coords_input: Optional[dict] 
            List with transforms to apply.
                point_coords (np.ndarray or None): A Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels.
                point_labels (np.ndarray or None): A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.
    """
        super().__init__(readers, transforms)
        # self.transform_coords_input = transform_coords_input
        self.multimask_output = multimask_output

        assert (
            len(self.readers) == 2
        ), "DatasetForSAM requires exactly 2 readers (image your label)"

        # assert (
        #     len(self.readers) == len(self.transforms)
        #     and len(self.transforms) == len(self.transform_coords_input)
        #     and len(self.readers) == len(self.transform_coords_input)
        # ), "DatasetForSAM requires exactly iquals lens (readers, transforms and transform_coords_input)"
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data and return data with SAM format (dict), where dict has:
        'image' (required): The image as a torch tensor in 3xHxW format.
        'label' (required): The label of the image.
        'original_size' (required): The original size of the image before transformation.
        'point_coords' (optional): (torch.Tensor) Batched point prompts for this image, with shape BxNx2. Already transformed to the input frame of the model.
        'point_labels' (optional): (torch.Tensor) Batched labels for point prompts, with shape BxN. (0 is background, 1 is object and -1 is pad)
        'boxes' (optional): (torch.Tensor) Batched box inputs, with shape Bx4.  Already transformed to the input frame of the model.
        'mask_inputs' (optional): (torch.Tensor) Batched mask inputs to the model, in the form Bx1xHxW.
        """

        data_readers = []
        for reader, transform in zip(self.readers, self.transforms):
            sample = reader[index]
            if transform is not None:
                sample = transform(sample)
            data_readers.append(sample)
        
        data = {}
        # apply transform_coords_input to image (only in the image, not in label)
        # if self.transform_coords_input['point_coords'] is not None: # TODO adicionar essa parte quando implementar treino com prompts
        # image = self.readers[0][index]
        # TODO Implementar algum script que coloque pontos aleatoriamente nas fácies
        # point_coords = self.transform_coords_input['point_coords'].apply_coords(point_coords, self.original_size)
        # coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        # labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        
        data['image'] = data_readers[0]
        data['label'] = data_readers[1]
        data['original_size'] = (int(data_readers[0].shape[1]), int(data_readers[0].shape[2])) # (tem que usar o shape depois do transform, se não dá erro) (int(image.shape[0]), int(image.shape[1]))
        data['multimask_output'] = self.multimask_output
        # TODO OBS: Só pode passar esses pontos se aplicar o transform_coords. Se tentar passar como None vai dar erro no Dataloader.
        # data['point_coords'] = None
        # data['point_labels'] = None
        # data['boxes'] = None
        # data['mask_inputs'] = None

        return data # (data, self.multimask_output)

""" class for create data module """
class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        transforms: _Transform = None,
        transform_coords_input: _Transform = None,
        multimask_output:bool = True,
        batch_size: int = 1,
        data_ratio: float = 1.0,
        num_workers: int = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms
        self.transform_coords_input = transform_coords_input
        self.multimask_output = multimask_output
        self.batch_size = batch_size
        self.data_ratio = data_ratio
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.datasets = {}

    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = TiffReader(self.train_path / "train")
            train_label_reader = PNGReader(self.annotations_path / "train")

            # applying ratio
            num_train_samples = int(len(train_img_reader) * self.data_ratio)
            if num_train_samples < len(train_img_reader):
                indices = random.sample(range(len(train_img_reader)), num_train_samples)
                train_img_reader = [train_img_reader[i] for i in indices]
                train_label_reader = [train_label_reader[i] for i in indices]
            
            train_dataset = DatasetForSAM(
                readers=[train_img_reader, train_label_reader],
                transforms=self.transforms,
                transform_coords_input=self.transform_coords_input,
                multimask_output=self.multimask_output
            )

            val_img_reader = TiffReader(self.train_path / "val")
            val_label_reader = PNGReader(self.annotations_path / "val")
            val_dataset = DatasetForSAM(
                readers=[val_img_reader, val_label_reader],
                transforms=self.transforms,
                transform_coords_input=self.transform_coords_input,
                multimask_output=self.multimask_output
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReader(self.train_path / "test")
            test_label_reader = PNGReader(self.annotations_path / "test")
            test_dataset = DatasetForSAM(
                readers=[test_img_reader, test_label_reader],
                transforms=self.transforms,
                transform_coords_input=self.transform_coords_input,
                multimask_output=self.multimask_output
            )
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def custom_collate_fn(self, batch):
        """
        Custom collate function for DataLoader to return a list of dictionaries.
        """
        return batch 

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.custom_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.custom_collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.custom_collate_fn
        )