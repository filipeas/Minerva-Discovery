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

from skimage.morphology import skeletonize

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

"""
# version 1 of dataset class for using in experiment 3.
- this version 1 implement the following features:
1. segment separately facies of seismic data, however, using prompts
2. the prompts is all sparses, specificaly all are positive points
3. its is used 3 positive points, equidistantly separated
"""
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

"""
# version 2 of dataset class for using in experiment 3.
- this version 2 implement the following features:
1. segment separately facies of seismic data, however, using prompts
2. the prompts is all sparses, all prompts is points (positives and negatives)
3. it is used skeleton of facie for set N positive points. to add negative points, is used a grid with
    equidistantly separated points.
"""
class DatasetForSAM_experiment_3_v2(SimpleDataset):
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
            
            # normalize and add 3 channels
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

                point_coords_positive, point_coords_negative = self.get_points_in_region(region=region, num_points_positive=self.num_points)
                self.samples.append((image, region, point_coords_positive, point_coords_negative))

    def get_points_in_region(self, region, num_points_positive=3, num_points_negative=7, offset=9, negative_method='grid'):
        """
        Processa a região binária e retorna os pontos positivos e negativos.
        
        Parâmetros:
            region (np.ndarray): Imagem binária (0 e 1) representando a região de interesse.
            num_points_positive (int): Número de pontos positivos a serem selecionados (usando o esqueleto).
            num_points_negative (int): Número de pontos negativos a serem calculados.
            offset (int): Margem para definir a região permitida para os pontos negativos.
            negative_method (str): Método para distribuir os pontos negativos. Opções: 'grid', 'poisson' ou 'thomas'.
            
        Retorna:
            tuple: (point_coords_positives, negative_points)
                - point_coords_positives (np.ndarray): Array (N, 2) com as coordenadas (x, y) dos pontos positivos.
                - negative_points (np.ndarray): Array (M, 2) com as coordenadas (x, y) dos pontos negativos.
        """
        # Garantir que a matriz tenha apenas duas dimensões
        if region.ndim == 3 and region.shape[0] == 1:
            region = region.squeeze(0)  # (1, H, W) -> (H, W)
        
        # Verificar se a região contém apenas valores 0 e 1
        unique_values = np.unique(region)
        if not (np.array_equal(unique_values, [0, 1]) or
                np.array_equal(unique_values, [1]) or
                np.array_equal(unique_values, [0])):
            raise ValueError(f"A matriz 'region' contém valores inesperados: {unique_values}. Esperado apenas 0 e 1.")
        
        # Obtém o esqueleto da região para os pontos positivos
        skeleton = skeletonize(region)
        y_indices, x_indices = np.where(skeleton == 1)
        
        # Se não houver pontos no esqueleto, retorna arrays vazios
        if len(y_indices) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        
        # Criar lista de pontos (em (x, y))
        skeleton_points = list(zip(x_indices, y_indices))
        
        # Calcular o centro da região branca usando os pixels com valor 1
        white_pixels = np.argwhere(region == 1)  # (y, x)
        overall_center = np.array([np.mean(white_pixels[:, 1]), np.mean(white_pixels[:, 0])])  # (x, y)
        
        # Ordenar pontos positivos pela distância ao centro
        skeleton_points = sorted(skeleton_points, key=lambda coord: np.linalg.norm(np.array(coord) - overall_center))
        
        # Selecionar pontos positivos equidistantes
        num_points_positive = min(num_points_positive, len(skeleton_points))
        indices_positive = np.linspace(0, len(skeleton_points) - 1, num_points_positive, dtype=int)
        selected_points_positive = [skeleton_points[i] for i in indices_positive]
        
        # Filtrar pontos que realmente pertencem à região branca (valor 1)
        point_coords_positives = np.array([
            (int(x), int(round(y)))
            for x, y in selected_points_positive
            if region[int(round(y)), int(x)] == 1
        ])
        
        # --- Cálculo dos pontos negativos usando método de distribuição ---
        H, W = region.shape
        # Define a região permitida para os negativos (evitando as bordas da imagem)
        x_min, x_max = offset, W - offset
        y_min, y_max = offset, H - offset
        
        # Se a imagem for muito pequena, ajusta a região permitida
        if x_min >= x_max or y_min >= y_max:
            x_min, y_min = 0, 0
            x_max, y_max = W - 1, H - 1

        # Seleciona o método de distribuição dos pontos negativos
        if negative_method == 'grid':
            # Gera pontos negativos apenas na região com pixel 0, com boa distribuição e longe das bordas.
            neg_pts = self.negative_points_grid_background(region, num_points_negative, offset)
        elif negative_method == 'poisson':
            raise NotImplemented()
        elif negative_method == 'thomas':
            raise NotImplemented()
        else:
            raise ValueError(f"Método de pontos negativos desconhecido: {negative_method}")
        
        # Ordenar pontos negativos pela distância ao centro geral da região branca
        # neg_pts = sorted(neg_pts, key=lambda coord: np.linalg.norm(np.array(coord) - overall_center))
        negative_points = np.array(neg_pts)
        
        return point_coords_positives, negative_points
    
    def negative_points_grid_background(self, region, num_points_negative, offset):
        """
        Gera uma grade central de pontos negativos apenas na região de fundo (pixels 0),
        espaçando os pontos para garantir que não fiquem próximos.
        
        Os pontos são distribuídos começando pelo centro da região permitida.
        """
        H, W = region.shape
        x_min, x_max = offset, W - offset - 1
        y_min, y_max = offset, H - offset - 1

        # Define o tamanho da célula para garantir espaçamento adequado
        step_x = (x_max - x_min) // int(np.sqrt(num_points_negative) + 1)
        step_y = (y_max - y_min) // int(np.sqrt(num_points_negative) + 1)

        points = []
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2  # Centro da região permitida

        # Gera a grade a partir do centro, espalhando para as extremidades
        for j in range(-num_points_negative // 2, num_points_negative // 2 + 1):
            for i in range(-num_points_negative // 2, num_points_negative // 2 + 1):
                x = int(cx + i * step_x)
                y = int(cy + j * step_y)

                if x_min <= x <= x_max and y_min <= y <= y_max and region[y, x] == 0:
                    points.append((x, y))

                if len(points) == num_points_negative:
                    return points

        return points
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        index: Tuple:
            - (image, label, point_coords)
        """
        image, label, point_coords_positive, point_coords_negative = self.samples[index]
        
        # preparing points and labels positives for add with prompt to SAM
        points_positives = [[x, y] for (x, y) in point_coords_positive]
        labels_positives = [1] * len(points_positives)

        # preparing points and labels negative for add with prompt to SAM
        points_negatives = [[x, y] for (x, y) in point_coords_negative]
        labels_negatives = [0] * len(points_negatives)

        # concatenate points positives and negatives
        combined_points = points_positives + points_negatives
        combined_labels = labels_positives + labels_negatives

        # convert to tensors
        points = torch.tensor(combined_points, dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(combined_labels, dtype=torch.long).unsqueeze(0)

        # image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        original_size = (int(image.shape[1]), int(image.shape[2]))

        # Verificar se original_size é uma tupla com 2 elementos
        if not isinstance(original_size, tuple) or len(original_size) != 2:
            raise ValueError(f"original_size is not a valid tuple: {original_size}")
        
        data = {
            'image': image,
            'label': label,
            'original_size': original_size,
            'point_coords': points,
            'point_labels': labels
        }

        return data

"""
# version 3 of dataset class for using in experiment 3.
- this version 3 implement the following features:
1. segment separately facies of seismic data, however, using prompts
2. the prompts is all sparses, all prompts is points (positives and negatives)
3. it is used a percentage of the areas for add points: for positive points 
    is used X% * 4 of the area of interest and then the points are equidistantly separated using a grid.
    the same is maked in background, using X% of the area of the background. ps: X% is a parameter defined by user.
"""
class DatasetForSAM_experiment_3_v3(SimpleDataset):
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
            
            # normalize and add 3 channels. process image
            image = data_readers[0]
            if image.shape[0] == 1: # If grayscale, repeat to create 3 channels
                image = np.repeat(image, 3, axis=0)
            image = np.clip(image * 255, 0, 255).astype(np.uint8) # scale to [0,255]
            image = torch.from_numpy(image).float() # converto to tensor

            # process label
            label = data_readers[1]
            if label.ndim == 2:  # If it's (H, W), add the channel dimension
                label = np.expand_dims(label, axis=0)
            label = torch.from_numpy(label).long() # Convert to tensor
            
            num_facies = np.unique(label)
            
            for facie in num_facies:
                region = np.zeros_like(label, dtype=np.uint8) # [H,W]
                region[label == facie] = 1

                point_coords_positive, point_coords_negative = self.get_points_in_region(
                    region=region, 
                    positive_percent=self.num_points * 4, 
                    negative_percent=self.num_points
                )
                self.samples.append((image, region, point_coords_positive, point_coords_negative))

    def get_points_in_region(self, region, positive_percent=50, negative_percent=50):
        """
        Processa a região binária e retorna os pontos positivos e negativos 
        com base na porcentagem informada.
        
        Parâmetros:
            region (np.ndarray): Imagem binária (0 e 1) representando a região de interesse.
            positive_percent (float): Porcentagem para definir a densidade da grade na região positiva (pixel 1).
            negative_percent (float): Porcentagem para definir a densidade da grade na região negativa (pixel 0).
            
        Retorna:
            tuple: (points_positive, points_negative)
                - points_positive (np.ndarray): Array (N, 2) com as coordenadas (x, y) dos pontos positivos.
                - points_negative (np.ndarray): Array (M, 2) com as coordenadas (x, y) dos pontos negativos.
        """
        # Garantir que a matriz seja 2D
        if region.ndim == 3 and region.shape[0] == 1:
            region = region.squeeze(0)
        
        # Verifica se a região contém apenas 0 e 1
        unique_values = np.unique(region)
        if not (np.array_equal(unique_values, [0, 1]) or
                np.array_equal(unique_values, [1]) or
                np.array_equal(unique_values, [0])):
            raise ValueError(f"A matriz 'region' contém valores inesperados: {unique_values}. Esperado apenas 0 e 1.")
        
        # Gera a grade de pontos para a região positiva (pixel 1)
        points_positive = self._get_grid_points(region, target_value=1, percent=positive_percent)
        
        # Gera a grade de pontos para a região negativa (pixel 0)
        points_negative = self._get_grid_points(region, target_value=0, percent=negative_percent)
        
        return points_positive, points_negative

    def _get_grid_points(self, region, target_value, percent):
        """
        Gera uma grade uniforme de pontos sobre o bounding box dos pixels que possuem o valor target_value,
        com densidade definida pela porcentagem informada.
        
        Parâmetros:
            region (np.ndarray): Imagem binária.
            target_value (int): Valor alvo da região (0 ou 1).
            percent (float): Porcentagem que define a densidade da grade (ex.: 50 para 50%).
            
        Retorna:
            np.ndarray: Array (N, 2) com as coordenadas (x, y) dos pontos que pertencem à região target.
        """
        # Obter as coordenadas dos pixels que correspondem ao target_value
        coords = np.argwhere(region == target_value)
        if coords.size == 0:
            return np.empty((0, 2), dtype=int)
        
        # Determinar o bounding box dos pixels do target
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Calcula o número de pontos em cada direção com base na porcentagem
        # Garantindo que haja pelo menos 2 pontos para formar uma grade
        n_x = max(2, int(round((x_max - x_min + 1) * (percent / 100.0))))
        n_y = max(2, int(round((y_max - y_min + 1) * (percent / 100.0))))
        
        # Gera os valores de x e y de forma equidistante dentro do bounding box
        x_vals = np.linspace(x_min, x_max, n_x)
        y_vals = np.linspace(y_min, y_max, n_y)
        xv, yv = np.meshgrid(x_vals, y_vals)
        
        # Converte para inteiros (índices de pixels) e organiza os pontos em (x, y)
        grid_points = np.vstack([xv.ravel(), yv.ravel()]).T.astype(int)
        
        # Filtra para manter apenas os pontos que estão de fato na região target
        valid_points = [pt for pt in grid_points if region[pt[1], pt[0]] == target_value]
        
        return np.array(valid_points)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        index: Tuple:
            - (image, label, point_coords)
        """
        image, label, point_coords_positive, point_coords_negative = self.samples[index]
        
        # preparing points and labels positives for add with prompt to SAM
        points_positives = [[x, y] for (x, y) in point_coords_positive]
        labels_positives = [1] * len(points_positives)

        # preparing points and labels negative for add with prompt to SAM
        points_negatives = [[x, y] for (x, y) in point_coords_negative]
        labels_negatives = [0] * len(points_negatives)

        # concatenate points positives and negatives
        combined_points = points_positives + points_negatives
        combined_labels = labels_positives + labels_negatives

        # convert to tensors
        points = torch.tensor(combined_points, dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(combined_labels, dtype=torch.long).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)
        original_size = (int(image.shape[1]), int(image.shape[2]))

        # Verificar se original_size é uma tupla com 2 elementos
        if not isinstance(original_size, tuple) or len(original_size) != 2:
            raise ValueError(f"original_size is not a valid tuple: {original_size}")
        
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