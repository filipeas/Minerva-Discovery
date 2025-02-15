from minerva.data.datasets.supervised_dataset import SimpleDataset
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.transforms.transform import _Transform
from minerva.data.readers.reader import _Reader
from torch.utils.data import DataLoader
import lightning as L
import torch
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
import random
import os

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
    
class DatasetForSAM(SimpleDataset):
    def __init__(
            self, 
            readers: List[_Reader], 
            transforms: Optional[_Transform] = None,
            num_points:int=3
    ):
        super().__init__(readers, transforms)

        assert (
            len(self.readers) == 2
        ), "DatasetForSAM requires exactly 2 readers (image your label)"

        self.num_points = num_points
        self.samples = []
        self._preprocess_data()
    
    def _preprocess_data(self):
        for index in range(len(self.readers[0])):
            data_readers = []
            for reader, transform in zip(self.readers, self.transforms):
                sample = reader[index]
                if transform is not None:
                    sample = transform(sample)
                data_readers.append(sample)
            
            image = data_readers[0]
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            image = (image * 255).clamp(0, 255).to(torch.uint8)
            label = data_readers[1]
            
            num_facies = np.unique(label)
            
            for facie in num_facies:
                region = np.zeros_like(label, dtype=np.uint8) # [H,W]
                region[label == facie] = 1

                point_coords = self.get_points_in_region(region=region, num_points=self.num_points)
                self.samples.append((image, region, point_coords))

    def get_points_in_region(self, region, num_points=3):
        # # Garantir que a região tem apenas valores 0 e 1
        # region = (region > 0).astype(np.uint8)

        # Garantir que a matriz tem apenas duas dimensões removendo a dimensão extra
        if region.ndim == 3 and region.shape[0] == 1:
            region = region.squeeze(0)  # Remove a primeira dimensão (1, H, W) -> (H, W)


        # Verificar se a região contém apenas valores 0 e 1
        unique_values = np.unique(region)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [1]) and not np.array_equal(unique_values, [0]):
            raise ValueError(f"A matriz 'region' contém valores inesperados: {unique_values}. Esperado apenas 0 e 1.")

        # Obter todas as coordenadas (y, x) da região branca
        y_indices, x_indices = np.where(region == 1)

        # Se não houver pontos na região, retornar uma lista vazia
        if len(y_indices) == 0:
            return []

        # Encontrar o centro vertical de cada coluna
        unique_x = np.unique(x_indices)
        central_y_coords = []

        for x in unique_x:
            y_in_column = y_indices[x_indices == x]

            if len(y_in_column) > 0:
                central_y = y_in_column[len(y_in_column) // 2]  # Pega um ponto real, não a média
                central_y_coords.append((x, central_y))

        # Ordenar os pontos pela coordenada x
        central_y_coords = sorted(central_y_coords, key=lambda coord: coord[0])

        # Selecionar pontos equidistantes
        num_points = min(num_points, len(central_y_coords))
        indices = np.linspace(0, len(central_y_coords) - 1, num_points, dtype=int)
        
        selected_points = [central_y_coords[i] for i in indices]

        # Filtrar pontos que realmente pertencem à região branca
        filtered_points = [(int(x), int(round(y)), 1) for x, y in selected_points if region[int(round(y)), int(x)] == 1]

        return filtered_points
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        index: Tuple:
            - (image, label, point_coords)
        """
        image, label, point_coords = self.samples[index]
        
        # preparing points and labels for add with prompt to SAM
        points = [[x, y] for (x, y, value) in point_coords]
        labels = [1] * len(points)

        # image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        original_size = (int(image.shape[1]), int(image.shape[2])) # torch.tensor((int(image.shape[1]), int(image.shape[2])), dtype=torch.long)

        # Verificar se original_size é uma tupla com 2 elementos
        if not isinstance(original_size, tuple) or len(original_size) != 2:
            raise ValueError(f"original_size is not a valid tuple: {original_size}")

        points = torch.tensor(points, dtype=torch.long).unsqueeze(0)  # Adicionando uma dimensão no início
        labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)  # Adicionando uma dimensão no início
        
        data = {
            'image': image,
            'label': label,
            'original_size': original_size,
            'point_coords': points,
            'point_labels': labels
        }

        return data

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        transforms: _Transform = None,
        num_points:int = 3,
        batch_size: int = 1,
        data_ratio: float = 1.0,
        filter_type: str = None,
        num_workers: int = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms
        self.num_points = num_points
        self.batch_size = batch_size
        self.data_ratio = data_ratio

        if filter_type not in (None, "il_", "xl_"):
            raise ValueError(f"filter_type must be 'il_', 'xl_', or None, but got '{filter_type}'")
        self.filter_type = filter_type

        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.datasets = {}

    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = TiffReader(self.train_path / "train")
            train_label_reader = PNGReader(self.annotations_path / "train")

            # applying filter for get only inline or crossline
            if self.filter_type:
                # to images
                train_img_reader.files = [
                    f for f in train_img_reader.files 
                    if f.name.startswith(self.filter_type) and f.name.lower().endswith((".tiff", ".tif"))
                ]
                # to labels
                train_label_reader.files = [
                    f for f in train_label_reader.files 
                    if f.name.startswith(self.filter_type) and f.name.lower().endswith((".png"))
                ]

            # applying ratio
            num_train_samples = int(len(train_img_reader) * self.data_ratio)
            if num_train_samples < len(train_img_reader):
                indices = random.sample(range(len(train_img_reader)), num_train_samples)
                train_img_reader = [train_img_reader[i] for i in indices]
                train_label_reader = [train_label_reader[i] for i in indices]
                
            train_dataset = DatasetForSAM(
                readers=[train_img_reader, train_label_reader],
                transforms=self.transforms,
                num_points=self.num_points
            )

            val_img_reader = TiffReader(self.train_path / "val")
            val_label_reader = PNGReader(self.annotations_path / "val")
            val_dataset = DatasetForSAM(
                readers=[val_img_reader, val_label_reader],
                transforms=self.transforms,
                num_points=self.num_points
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReader(self.train_path / "test")
            test_label_reader = PNGReader(self.annotations_path / "test")
            test_dataset = DatasetForSAM(
                readers=[test_img_reader, test_label_reader],
                transforms=self.transforms,
                num_points=self.num_points
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