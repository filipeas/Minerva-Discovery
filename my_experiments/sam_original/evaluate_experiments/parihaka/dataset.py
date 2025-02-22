import os
from pathlib import Path
import numpy as np
from typing import List, Optional, Tuple, Union
import lightning as L
import torch
from torch.utils.data import DataLoader
from functools import partial

from minerva.data.datasets.supervised_dataset import SimpleDataset
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.transforms.transform import _Transform
from minerva.data.readers.reader import _Reader
from minerva.utils.typing import PathLike
from minerva.data.data_modules.parihaka import (
    default_train_transforms,
    default_test_transforms,
)

# Partial functions for the TiffReader and PNGReader with numeric sort
# and delimiter "_" for the Parihaka dataset.
TiffReaderWithNumericSort = partial(
    TiffReader, sort_method=["text", "numeric"], delimiter="_", key_index=[0, 1]
)
PNGReaderWithNumericSort = partial(
    PNGReader, sort_method=["text", "numeric"], delimiter="_", key_index=[0, 1]
)

# Function for format the batch for list of dictionaries
def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to return a list of dictionaries.
    """
    return batch 

class DatasetForSAM_experiment_1(SimpleDataset):
    def __init__(
            self, 
            readers: List[_Reader], 
            transforms: Optional[_Transform] = None,
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
    """
        super().__init__(readers, transforms)
        self.multimask_output = multimask_output

        assert (
            len(self.readers) == 2
        ), "DatasetForSAM requires exactly 2 readers (image your label)"

        assert (
            len(self.readers) == len(self.transforms)
        ), "DatasetForSAM requires exactly iquals lens (readers and transforms)"
    
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
        
        # process image
        image = data_readers[0]
        if image.shape[0] == 1: # If grayscale, repeat to create 3 channels
            image = np.repeat(image, 3, axis=0)
        image = np.clip(image * 255, 0, 255).astype(np.uint8) # scale to [0,255]
        image = torch.from_numpy(image).float() # converto to tensor

        # process label
        label = data_readers[1]
        if label.ndim == 2:  # If it's (H, W), add the channel dimension
            label = np.expand_dims(label, axis=0)
        label = torch.from_numpy(label).long()  # Convert to tensor

        data = {
            'image': image,
            'label': label,
            'original_size': (int(data_readers[0].shape[1]), int(data_readers[0].shape[2])),
            'multimask_output': self.multimask_output
        }

        return data

class DatasetForSAM_experiment_2(SimpleDataset):
    def __init__(
            self, 
            readers: List[_Reader], 
            transforms: Optional[_Transform] = None,
            select_facie:int=0
    ):
        super().__init__(readers, transforms)
        self.select_facie = select_facie # pode ser: 0 (facie mais escura), a 5 (facie mais clara)

        assert (
            len(self.readers) == 2
        ), "DatasetForSAM requires exactly 2 readers (image your label)"
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        data_readers = []
        for reader, transform in zip(self.readers, self.transforms):
            sample = reader[index]
            if transform is not None:
                sample = transform(sample)
            data_readers.append(sample)

        # process image
        image = data_readers[0]
        if image.shape[0] == 1: # If grayscale, repeat to create 3 channels
            image = np.repeat(image, 3, axis=0)
        image = np.clip(image * 255, 0, 255).astype(np.uint8) # scale to [0,255]
        image = torch.from_numpy(image).float() # converto to tensor

        # process label
        label = data_readers[1]
        if label.ndim == 2:  # If it's (H, W), add the channel dimension
            label = np.expand_dims(label, axis=0)
        label = torch.from_numpy(label).long()  # Convert to tensor

        # Gera uma máscara binária apenas para a fácie selecionada
        binary_mask = (label == self.select_facie).to(torch.uint8)

        data = {
            'image': image,
            'label': binary_mask,
            'original_size': (int(image.shape[1]), int(image.shape[2])),
            'class_id': self.select_facie
        }

        return data

class DatasetForSAM_experiment_3(SimpleDataset):
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

            # process image
            image = data_readers[0]
            if image.shape[0] == 1: # If grayscale, repeat to create 3 channels
                image = np.repeat(image, 3, axis=0)
            image = np.clip(image * 255, 0, 255).astype(np.uint8) # scale to [0,255]
            image = torch.from_numpy(image).float() # converto to tensor

            # process label
            label = data_readers[1]
            if label.ndim == 2:  # If it's (H, W), add the channel dimension
                label = np.expand_dims(label, axis=0)
            label = torch.from_numpy(label).long()  # Convert to tensor
            
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

class ParihakaDataModuleForSAM(L.LightningDataModule):
    """Default data module for the Parihaka dataset. This data module creates a
    supervised reconstruction dataset for training, validation, testing, and
    prediction with default transforms to read the images and labels.

    The parihaka dataset is organized as follows:
    root_data_dir
    ├── train
    │   ├── il_1.tif
    |   ├── il_2.tif
    |   ├── ...
    ├── val
    │   ├── il_1.tif
    |   ├── il_2.tif
    |   ├── ...
    ├── test
    │   ├── il_1.tif
    |   ├── il_2.tif
    |   ├── ...
    root_annotation_dir
    ├── train
    │   ├── il_1.png
    |   ├── il_2.png
    |   ├── ...
    ├── val
    │   ├── il_1.png
    |   ├── il_2.png
    |   ├── ...
    ├── test
    │   ├── il_1.png
    |   ├── il_2.png
    |   ├── ...

    The `root_data_dir` contains the seismic images and the
    `root_annotation_dir` contains the corresponding labels. Files with the
    same name in the same directory are assumed to be pairs of seismic images
    and labels. For instance `root_data_dir/train/il_1.tif` and
    `root_annotation_dir/train/il_1.png` are assumed to be a pair of seismic
    image and label.

    Original parihaka dataset contains inlines and crosslines in train and val
    directories. Inlines have dimensions (1006, 590, 3) and crosslines have
    dimensions (1006, 531, 3). By default, crosslines are padded to (1006, 590)
    and all images are transposed to (3, 1006, 590) format. Labels are also
    padded to (1, 1006, 590) and are not transposed. Finally, images are cast to
    float32 and labels are cast to int32.
    """

    def __init__(
        self,
        dataset_class:type,
        dataset_params: dict,
        root_data_dir: PathLike,
        root_annotation_dir: PathLike,
        train_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        valid_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        test_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        drop_last: bool = True,
        data_loader_kwargs: Optional[dict] = None,
    ):
        """Initialize the ParihakaDataModule with the root data and annotation
        directories. The data module is initialized with default training and
        testing transforms.

        Parameters
        ----------
        root_data_dir : str
            Root directory containing the seismic images. Inside this directory
            should be subdirectories `train`, `val`, and `test` containing the
            training, validation, and testing TIFF images.
        root_annotation_dir : str
            Root directory containing the annotations. Inside this directory
            should be subdirectories `train`, `val`, and `test` containing the
            training, validation, and testing PNG annotations. Files with the
            same name in the same directory are assumed to be pairs of seismic
            images and labels.
        train_transforms : Optional[Union[_Transform, List[_Transform]]], optional
            2-element list of transform pipelines for the image and label reader.
            Transforms to apply to the training and validation datasets. If
            None, default training transforms are used, which pads images to
            (1, 1006, 590) and transposes them to (3, 1006, 590) format. Labels
            are also padded to (1006, 590). By default None
        valid_transforms: Optional[Union[_Transform, List[_Transform]]], optional
            2-element list of transform pipelines for the image and label reader.
            Transforms to apply to the validation datasets. If None, default
            training transforms are used, which pads images to (1006, 590) and
            transposes them to (3, 1006, 590) format. Labels are also padded to
            (1, 1006, 590). By default None
        test_transforms : Optional[Union[_Transform, List[_Transform]]], optional
            2-element list of transform pipelines for the image and label reader.
            Transforms to apply to the testing and prediction datasets. If None,
            default testing transforms are used, which transposes images to
            CxHxW format. Labels are untouched. By default None
        batch_size : int, optional
            Default batch size for the dataloaders, by default 1
        num_workers : Optional[int], optional
            Number of workers for the dataloaders, by default None. If None,
            the number of workers is set to the number of CPUs on the system.
        drop_last : bool, optional
            Whether to drop the last batch if it is smaller than the batch size,
            by default True.
        data_loader_kwargs : Optional[dict], optional
            Aditional keyword arguments to pass to the DataLoader instantiation,
            for training, validation, testing, and prediction dataloaders.
            By default None. Note that, `batch_size`, `num_workers`, and 
            `drop_last` are ignored if passed in this dictionary, as they are
            already presented in the ParihakaDataModule constructor.
        """
        super().__init__()
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        self.root_data_dir = Path(root_data_dir)
        self.root_annotation_dir = Path(root_annotation_dir)
        self.train_transforms = train_transforms or default_train_transforms()
        self.valid_transforms = valid_transforms or default_train_transforms()
        self.test_transforms = test_transforms or default_test_transforms()
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.drop_last = drop_last
        self.datasets = {}
        
        self.data_loader_kwargs = data_loader_kwargs or {}
        # Update the data loader kwargs with the batch size, num workers, and
        # drop last parameters, passed to the ParihakaDataModule.
        self.data_loader_kwargs.update(
            {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "drop_last": self.drop_last,
                "collate_fn": custom_collate_fn
            }
        )
        # Remove the shuffle parameter from the data loader kwargs, as it is
        # handled by the dataloaders.
        self.data_loader_kwargs.pop("shuffle", None)

    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = TiffReaderWithNumericSort(
                self.root_data_dir / "train"
            )
            train_label_reader = PNGReaderWithNumericSort(
                self.root_annotation_dir / "train",
            )
            train_dataset = self.dataset_class(
                readers=[train_img_reader, train_label_reader],
                transforms=self.train_transforms,
                **self.dataset_params
            )

            val_img_reader = TiffReaderWithNumericSort(
                self.root_data_dir / "val",
            )
            val_label_reader = PNGReaderWithNumericSort(
                self.root_annotation_dir / "val",
            )
            val_dataset = self.dataset_class(
                readers=[val_img_reader, val_label_reader],
                transforms=self.valid_transforms,
                **self.dataset_params
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReaderWithNumericSort(
                self.root_data_dir / "test",
            )
            test_label_reader = PNGReaderWithNumericSort(
                self.root_annotation_dir / "test",
            )
            test_dataset = self.dataset_class(
                readers=[test_img_reader, test_label_reader],
                transforms=self.test_transforms,
                **self.dataset_params
            )
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _get_dataloader(self, partition: str, shuffle: bool):
        return DataLoader(
            self.datasets[partition],
            shuffle=shuffle,
            **self.data_loader_kwargs
        )

    def train_dataloader(self):
        return self._get_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader("val", shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader("test", shuffle=False)

    def predict_dataloader(self):
        return self._get_dataloader("predict", shuffle=False)

    def __str__(self) -> str:
        return f"""DataModule
    Data: {self.root_data_dir}
    Annotations: {self.root_annotation_dir}
    Batch size: {self.batch_size}"""

    def __repr__(self) -> str:
        return str(self)