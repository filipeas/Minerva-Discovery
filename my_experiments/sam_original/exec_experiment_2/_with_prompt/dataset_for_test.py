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
from torchmetrics import JaccardIndex
import pandas as pd
import cv2
import random
import os
from tqdm import tqdm

class DataModule_for_AUC(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        transforms: _Transform = None,
        batch_size: int = 1,
        data_ratio: float = 1.0,
        num_workers: int = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms
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
                
            train_dataset = DatasetForSAM_for_AUC(
                readers=[train_img_reader, train_label_reader],
                transforms=self.transforms
            )

            val_img_reader = TiffReader(self.train_path / "val")
            val_label_reader = PNGReader(self.annotations_path / "val")
            val_dataset = DatasetForSAM_for_AUC(
                readers=[val_img_reader, val_label_reader],
                transforms=self.transforms
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReader(self.train_path / "test")
            test_label_reader = PNGReader(self.annotations_path / "test")
            test_dataset = DatasetForSAM_for_AUC(
                readers=[test_img_reader, test_label_reader],
                transforms=self.transforms
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

class DatasetForSAM_for_AUC(SimpleDataset):
    def __init__(
            self, 
            readers: List[_Reader], 
            transforms: Optional[_Transform] = None,
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
        
        data = {
            'image': data_readers[0],
            'label': data_readers[1],
            'original_size': (int(data_readers[0].shape[1]), int(data_readers[0].shape[2]))
        }
    
        return data

class AUC_calculate_v1():
    def __init__(self,
                 model,
                 dataloader,
                 save_dir,
                 multimask_output:bool=False,
                 num_points:int=3,
                 experiment_num:int=0,
                 ):
        self.model = model
        self.dataloader = dataloader
        self.num_points = num_points
        self.save_dir = save_dir
        self.multimask_output = multimask_output
        self.experiment_num = experiment_num
        self.set_points() # reseta pontos

        self.miou_metric = JaccardIndex(task="multiclass", num_classes=2).to(self.model.device) # binario
    
    def set_points(self):
        # Inicialize os acumuladores como arrays vazios
        self.accumulated_coords = np.empty((0, 2), dtype=int)  # Nx2 array
        self.accumulated_labels = np.empty((0,), dtype=int)   # Array de comprimento N
    
    def process(self):
        self.results = pd.DataFrame(columns=['sample_id', 'facie_id', 'accumulated_point', 'iou', 'num_points'])

        # for batch_idx, batch in enumerate(self.dataloader):
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Processando batches")):
            for item in batch:
                img = item['image']
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                img = (img * 255).clamp(0, 255).to(torch.uint8)
                image = img.to(self.model.device)
                label = item['label'].squeeze(0).numpy()#.to(self.model.device)
                # plt.figure(figsize=(6, 6))  # Define o tamanho da figura
                # plt.imshow(label, cmap='gray')  # Mostra a imagem, com escala de cinza (ou RGB se for colorida)
                # plt.title("label")  # Adiciona o título
                # plt.axis('off')  # Desliga os eixos para uma visualização limpa
                # plt.show()
                # break

                num_facies = np.unique(label) # num de facies da amostra
                point_type = 'positive' # inicia com ponto positivo, depois pode mudar para 'negative'

                for i, facie in enumerate(num_facies):
                    region = np.zeros_like(label, dtype=np.uint8) # [H,W]
                    region[label == facie] = 1
                    real_label = region
                    # plt.figure(figsize=(6, 6))  # Define o tamanho da figura
                    # plt.imshow(region, cmap='gray')  # Mostra a imagem, com escala de cinza (ou RGB se for colorida)
                    # plt.title("region")  # Adiciona o título
                    # plt.axis('off')  # Desliga os eixos para uma visualização limpa
                    # plt.show()
                    # break

                    for point in range(self.num_points):
                        # calculando centro da regiao e retornando sua coordenada
                        point_coords, point_labels = self.calculate_center_region(region=region, point_type=point_type)
                    
                        # definindo amostra (batch) a ser inserido no modelo
                        batch = {
                            'image': image,
                            'label': label,
                            'original_size': (int(image.shape[1]), int(image.shape[2])),
                            'point_coords': torch.tensor(point_coords, dtype=torch.long).unsqueeze(0),
                            'point_labels': torch.tensor(point_labels, dtype=torch.long).unsqueeze(0)
                        }
                    
                        # Inferência
                        outputs = self.model([batch], multimask_output=self.multimask_output) # batch tem que ser uma lista de dict. multimask_output é dado no init do model
                    
                        # calcular IoU
                        gt_tensor = torch.tensor(real_label).to(self.model.device)  # Converta para tensor 2D e mova para GPU
                        pred_tensor = torch.tensor(outputs[0]['masks'].squeeze()).to(self.model.device)  # Remover a dimensão extra e mover para GPU
                        
                        iou_score = self.miou_metric(pred_tensor, gt_tensor)

                        # print("real_label shape: ", real_label.shape)
                        # print('pred shape: ', outputs[0]['masks'].squeeze().numpy().shape)
                        diff, new_point_type = self.calculate_diff_label_pred(label=real_label, pred=outputs[0]['masks'].squeeze().numpy())
                    
                        # salvando progresso
                        new_row = pd.DataFrame([{
                            'sample_id': batch_idx,
                            'facie_id': facie,
                            'accumulated_point': point + 1,
                            'iou': iou_score.item(),
                            'num_points': self.num_points
                        }])

                        self.results = pd.concat([self.results, new_row], ignore_index=True)
                    
                        # plot experimental a cada 50 amostras
                        # if idx % 50 == 0:
                        # plot_all(
                        #     image=image.permute(1, 2, 0),
                        #     label=real_label,
                        #     pred=outputs[0]['masks'].squeeze().numpy(),
                        #     diff=diff,
                        #     score=iou_score,
                        #     point_coords=self.accumulated_coords,
                        #     point_labels=self.accumulated_labels
                        # )
                        region = diff # [H,W], atualiza para a proxima regiao
                        point_type = new_point_type # 'positive' ou 'negative', atualiza para a proxima regiao
                        # break # testa só 1 ponto
                    point_type = 'positive' # reinicia tipo do primeiro ponto (sempre deve ser positivo o primeiro)
                    self.set_points() # reinicia empilhamento de pontos
                    # break # testa só 1 facie
                # break # testar só uma amostra
            # break # testar só uma amostra
        
        self.results.to_csv(f'{self.save_dir}/iou_results_{self.experiment_num}.csv', index=False)

    def calculate_center_region(self, region: np.array, point_type: str, min_distance: int = 10):
        """
        Calcula o centroide da maior região de pixels brancos de uma imagem binária,
        deslocando horizontalmente o ponto se ele estiver próximo demais dos acumulados.

        Args:
            region (np.array): Imagem binária com a região de interesse (pixels brancos).
            point_type (str): Tipo do ponto ('positive' ou 'negative').
            min_distance (int): Distância mínima permitida entre pontos.

        Returns:
            point_coords (np.ndarray): Array Nx2 de pontos acumulados.
            point_labels (np.ndarray): Array N de rótulos acumulados.
        """
        if not isinstance(region, np.ndarray):
            raise TypeError("region needs to be a NumPy array.")
        
        # Encontrar as componentes conectadas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)

        if num_labels < 2:  # Apenas fundo e nenhuma região branca
            raise ValueError("No connected white regions found in the binary image.")
        
        # Ignorar o rótulo 0 (fundo), pegar a maior componente conectada
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[largest_label]
        center_x, center_y = int(center_x), int(center_y)
        new_coords = np.array([[center_x, center_y]])

        # Verificar se o ponto está muito próximo dos anteriores
        if self.accumulated_coords.shape[0] > 0:
            distances = np.sqrt(np.sum((self.accumulated_coords - new_coords) ** 2, axis=1))
            if np.any(distances < min_distance):
                # print("Ponto muito próximo ao anterior, deslocando horizontalmente...")
                
                # Tentar deslocar o ponto horizontalmente dentro da região branca
                region_height, region_width = region.shape
                # Tenta deslocar horizontalmente usando o min_distance
                for delta_x in range(min_distance, region_width, min_distance):  # Incrementa em min_distance
                    candidate_x_right = center_x + delta_x
                    candidate_x_left = center_x - delta_x

                    # Verifica primeiro para direita, depois para esquerda
                    if candidate_x_right < region_width and region[center_y, candidate_x_right] > 0:
                        center_x = candidate_x_right
                        break
                    elif candidate_x_left >= 0 and region[center_y, candidate_x_left] > 0:
                        center_x = candidate_x_left
                        break

                new_coords = np.array([[center_x, center_y]])

        # Definir o rótulo (positivo ou negativo)
        if point_type == 'positive':
            new_labels = np.array([1])
        elif point_type == 'negative':
            new_labels = np.array([0])
        else:
            raise ValueError("Invalid point_type. Must be 'positive' or 'negative'.")

        # Acumular os resultados
        self.accumulated_coords = np.vstack([self.accumulated_coords, new_coords])
        self.accumulated_labels = np.hstack([self.accumulated_labels, new_labels])

        return self.accumulated_coords, self.accumulated_labels
    
    def calculate_diff_label_pred(self, label:np.array, pred:np.array):
        """
        Calcula a diferença entre duas imagens binárias e determina se a área externa ou interna é maior.

        Args:
            label (np.array): Imagem binária de referência (label).
            pred (np.array): Imagem binária predita (pred).

        Returns:
            diff_colored (np.array): Imagem colorida representando as diferenças.
            point_type (str): 'negative' se a área externa for maior, 'positive' se a interna for maior.
        """
        if label.shape != pred.shape:
            raise ValueError("Label and Pred images have differents shapes. Check it before call calculate_dif_label_pred() function.")

        # Máscaras para regiões de diferença
        mask_outward = (label > pred)  # Diferença para fora -> Vermelho
        mask_inward = (label < pred)  # Diferença para dentro -> Azul

        area_outward = np.sum(mask_outward)
        area_inward = np.sum(mask_inward)

        diff_binary = teste1 = teste2 = np.zeros(label.shape, dtype=np.uint8) # [H,W]

        # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # teste1[mask_outward] = 1
        # axes[0].imshow(teste1)
        # axes[0].set_title('Image 1')
        # axes[0].axis('off')
        # teste2[mask_inward] = 1
        # axes[1].imshow(teste2)
        # axes[1].set_title('Image 2')
        # axes[1].axis('off')
        # plt.tight_layout()
        # plt.show()

        # Comparar as áreas
        if area_outward > area_inward:
            diff_binary[mask_outward] = 1
            point_type = 'positive'
        else:
            diff_binary[mask_inward] = 1
            point_type = 'negative'
        
        return diff_binary, point_type