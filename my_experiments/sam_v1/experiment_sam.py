import os
import random
import numpy as np
import gc
import time

from tqdm import tqdm
import argparse
import json
from datetime import datetime
import shutil
import psutil

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.models.nets.image.segment_anything.sam_lora import SAMLoRA
from minerva.transforms.transform import _Transform
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.data.datasets.base import SimpleDataset
from minerva.data.readers.reader import _Reader

from typing import List, Optional, Tuple

import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

def log_memory():
    # Uso de memória da CPU
    cpu_memory = psutil.virtual_memory()
    cpu_used_mb = cpu_memory.used / (1024 ** 2)  # Convertido para MB
    print(f"CPU Memory Usage: {cpu_memory.percent}% ({cpu_used_mb:.2f} MB)")
    
    # Uso de memória da GPU
    if torch.cuda.is_available():
        gpu_used_mb = torch.cuda.memory_allocated() / (1024 ** 2)  # Convertido para MB
        gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Total da GPU em MB
        print(f"GPU Memory Usage: {gpu_used_mb:.2f} MB / {gpu_total_mb:.2f} MB "
              f"({(gpu_used_mb / gpu_total_mb) * 100:.2f}%)")
    else:
        print("No GPU available.")

def _init_experiment(
        vit_model,
        checkpoint_path,
        train_path,
        annotation_path,
        config_path,
        name,
        image_size=512,
        num_classes=5,
        batch_size=2,
        alpha=1,
        rank=4,
        N=10, 
        epochs=500, 
        data_ratios=[0.01, 1.0],
        ):
    # Gerar um identificador único baseado no timestamp atual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Criar o diretório único para salvar os gráficos
    save_dir = Path(os.getcwd()) / f"results/{name}_experiment_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copiar o arquivo de configuração para o diretório de resultados
    if config_path:
        shutil.copy(config_path, save_dir / "config_copy.json")

    # Inicializar listas para armazenar os resultados
    results = {ratio: {'train_metrics': [], 'val_metrics': [], 'test_metrics': []} for ratio in data_ratios}

    # Loop pelos experimentos
    for ratio in data_ratios:
        print(f"Executando experimentos com {int(ratio*100)}% dos dados...")
        
        for _ in tqdm(range(N)):
            model = SAMLoRA(
                image_size=image_size,
                num_classes=num_classes,
                alpha=alpha,
                rank=rank,
                checkpoint=checkpoint_path,
                vit_model=vit_model
            )

            log_memory() # verificando consumo
            
            train_metrics, val_metrics, test_metrics = train_and_evaluate(name, image_size, model, ratio, epochs, batch_size=batch_size, train_path=train_path, annotation_path=annotation_path)
            # Armazene os resultados para cada experimento
            results[ratio]['train_metrics'].append(train_metrics)
            results[ratio]['val_metrics'].append(val_metrics)
            results[ratio]['test_metrics'].append(test_metrics)

            del model, train_metrics, val_metrics, test_metrics
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
    
    # Cálculo de média e desvio padrão para mIoU
    def extract_miou(metrics):
        mious = [metric['test_mIoU'] for experiment in metrics for metric in experiment]
        return mious
    
    # Cálculo de média e desvio padrão
    means = {ratio: np.mean(extract_miou(results[ratio]['test_metrics'])) for ratio in data_ratios}
    stds = {ratio: np.std(extract_miou(results[ratio]['test_metrics'])) for ratio in data_ratios}
    # print(f"results: {results}")
    # print(f"means: {means}")
    # print(f"stds: {stds}")
    
    matplotlib.use('Agg')

    # Gráfico de barra para mIoU de 1% e 100% dos dados
    plt.figure(figsize=(8, 6))
    plt.bar([f"{int(ratio*100)}% dos Dados" for ratio in data_ratios], 
            [means[ratio] for ratio in data_ratios], 
            yerr=[stds[ratio] for ratio in data_ratios], capsize=5)
    plt.ylabel('mIoU Médio')
    plt.title('Comparação de mIoU para 1% e 100% dos Dados')
    plt.savefig(save_dir / 'miou_comparison.png')
    # plt.close()
    
    def extract_metric(metrics, metric_name):
        # Extrai o valor de cada época para uma métrica específica
        return [[epoch_metrics[metric_name] for epoch_metrics in epoch] for epoch in metrics]

    for ratio in data_ratios:
        label = f"{int(ratio*100)}% dos Dados"
        
        # Calculando métricas para cada época
        train_loss = np.array(extract_metric(results[ratio]['train_metrics'], 'loss'))
        val_loss = np.array(extract_metric(results[ratio]['val_metrics'], 'loss'))
        train_mIoU = np.array(extract_metric(results[ratio]['train_metrics'], 'mIoU'))
        val_mIoU = np.array(extract_metric(results[ratio]['val_metrics'], 'mIoU'))
        # print(len(train_loss), train_loss)
        # print(len(val_loss), val_loss)
        # print(len(train_mIoU), train_mIoU)

        # Calcula média e desvio padrão em cada época
        train_loss_mean = np.mean(train_loss, axis=0)
        train_loss_std = np.std(train_loss, axis=0)
        val_loss_mean = np.mean(val_loss, axis=0) # remove o primeiro pq ele é antes de comecar o treino
        val_loss_std = np.std(val_loss, axis=0) # remove o primeiro pq ele é antes de comecar o treino
        train_mIoU_mean = np.mean(train_mIoU, axis=0)
        train_mIoU_std = np.std(train_mIoU, axis=0)
        val_mIoU_mean = np.mean(val_mIoU, axis=0)
        val_mIoU_std = np.std(val_mIoU, axis=0)

        epochs_range = np.arange(len(train_loss_mean))

        # Gráfico de perda com duas linhas
        # print(len(epochs_range), epochs_range)
        # print(len(val_loss_mean), val_loss_mean)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_loss_mean, label="Train Loss", color='blue')
        plt.fill_between(epochs_range, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color='blue', alpha=0.2)
        plt.plot(epochs_range, val_loss_mean, label="Validation Loss", color='orange')  # Adiciona linha de validação
        plt.fill_between(epochs_range, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, color='orange', alpha=0.2)  # Intervalo de confiança da validação
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.title(f"Loss no Treinamento e Validação - {label}")
        plt.legend()
        plt.savefig(save_dir / f'train_val_loss_{int(ratio*100)}.png')
        
        # Gráfico de mIoU
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_mIoU_mean, label=f"Train mIoU", color="blue")
        plt.fill_between(epochs_range, train_mIoU_mean - train_mIoU_std, train_mIoU_mean + train_mIoU_std, color="blue", alpha=0.2)
        plt.plot(epochs_range, val_mIoU_mean, label=f"Val mIoU", color="orange")
        plt.fill_between(epochs_range, val_mIoU_mean - val_mIoU_std, val_mIoU_mean + val_mIoU_std, color="orange", alpha=0.2)
        plt.xlabel("Épocas")
        plt.ylabel("mIoU")
        plt.title(f"mIoU no Treinamento e Validação - {label}")
        plt.legend()
        plt.savefig(save_dir / f'train_miou_{int(ratio*100)}.png')
        # plt.close()

# Função para treinar e testar o modelo
def train_and_evaluate(
        name,
        image_size,
        model, 
        ratio, 
        epochs, 
        train_path,
        annotation_path,
        batch_size=8,
        ):

    # Criando data module com o ratio definido
    data_module = PatchingModule(
        train_path=train_path,
        annotations_path=annotation_path,
        patch_size=image_size,
        stride=32,
        batch_size=batch_size,
        ratio=ratio,
    )

    # Define o callback para salvar o modelo com base no menor valor da métrica de validação
    current_date = datetime.now().strftime("%Y-%m-%d")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", # Métrica para monitorar
        dirpath="./checkpoints", # Diretório onde os checkpoints serão salvos
        filename=f"{name}_final_train-ratio-{ratio}-{current_date}-{{epoch:02d}}-{{val_loss:.2f}}", # Nome do arquivo do checkpoint
        save_top_k=1, # Quantos melhores checkpoints salvar (no caso, o melhor)
        mode="min", # Como a métrica deve ser tratada (no caso, 'min' significa que menor valor de val_loss é melhor)
    )
    
    logger = CSVLogger("logs", name="sam_model")

    # Inicialize o logger e o treinador para cada execução
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Inicialize o pipeline e execute o treinamento
    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        save_run_status=True
    )
    pipeline.run(data=data_module, task="fit")
    test_metrics = pipeline.run(data=data_module, task="test")

    # DEBUG: Extrair métricas (exemplo: mIoU e perda)
    # print(10*'-')
    # print("(train_metrics_acc): ", pipeline.model.train_metrics_acc)
    # print("(train_loss_acc): ", pipeline.model.train_loss_acc)
    # print("(val_metrics_acc): ", pipeline.model.val_metrics_acc)
    # print("(val_loss_acc): ", pipeline.model.val_loss_acc)
    # print("(test): ", test_metrics)
    # print(10*'-')
    train_metrics = pipeline.model.train_metrics_acc # todas as métricas de treino
    pipeline.model.train_metrics_acc = []
    # train_loss = pipeline.model.train_loss_acc['oi'] # todas as loss de treino
    val_metrics = pipeline.model.val_metrics_acc[1:] # todas as métricas de validacao
    pipeline.model.val_metrics_acc = []
    # val_loss = pipeline.model.val_loss_acc['oi'] # todas as loss de validacao

    del train_path, annotation_path, data_module, trainer, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    return train_metrics, val_metrics, test_metrics

class SupervisedDatasetPatches(SimpleDataset):
    def __init__(self, readers: List[_Reader], transforms: Optional[_Transform] = None, patch_size: int = 255, stride: int = 32):
        """Adds support for splitting images into patches.

        Parameters
        ----------
        readers: List[_Reader]
            List of data readers. It must contain exactly 2 readers.
            The first reader for the input data and the second reader for the
            target data.
        transforms: Optional[_Transform]
            Optional data transformation pipeline.
        patch_size: int
            Size of the patches into which the images will be divided.
        stride: int
            Stride used to extract patches from images.
        Raises
        -------
            AssertionError: If the number of readers is not exactly 2.
        """
        super().__init__(readers, transforms)
        self.patch_size = patch_size
        self.stride = stride
        self._patch_indices = []
        self._precompute_patch_indices()

        assert (
            len(self.readers) == 2
        ), "SupervisedReconstructionDataset requires exactly 2 readers"
    
    def _precompute_patch_indices(self):
        """Precomputes patch indices for all images."""
        for img_idx in range(len(self.readers[0])):
            # Obtem a dimensão da imagem para calcular o número de patches
            image = self.readers[0][img_idx]
            h, w = image.shape[:2]
            num_patches_h = (h - self.patch_size) // self.stride + 1
            num_patches_w = (w - self.patch_size) // self.stride + 1
            for patch_idx in range(num_patches_h * num_patches_w):
                self._patch_indices.append((img_idx, patch_idx))
    
    def _extract_single_patch(self, data, patch_idx, patch_size=255, stride=32, img_type='image'):
        if img_type == 'image': # caso seja imagens de entrada (h, w, c)
            h, w, _ = data.shape
        else: # caso seja labels de entrada (h, w)
            h, w = data.shape
        num_patches_w = (w - patch_size) // stride + 1
        row = patch_idx // num_patches_w # numero da linha do patch
        col = patch_idx % num_patches_w # numero da coluna do patch
        i, j = row * stride, col * stride # coordenada do patch no grid
        patch = data[i:i + patch_size, j:j + patch_size]
        if img_type == 'image':
            return patch.transpose(2, 0, 1).astype(np.float32) # (C H W)
        else:
            return patch.astype(np.int64)
    
    def __len__(self):
        """Returns the total number of patches."""
        return len(self._patch_indices)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load data and return a single patch."""
        img_idx, patch_idx = self._patch_indices[index]
        input_data = self.readers[0][img_idx]
        target_data = self.readers[1][img_idx]

        input_patch = self._extract_single_patch(input_data, patch_idx, img_type='image')
        target_patch = self._extract_single_patch(target_data, patch_idx, img_type='label')
        return input_patch, target_patch

class PatchingModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        patch_size: int = 255,
        stride: int = 32,
        batch_size: int = 8,
        ratio: int = 1.0,
        transforms: _Transform = None,
        num_workers: int = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms
        self.batch_size = batch_size
        self.ratio = ratio
        self.patch_size = patch_size
        self.stride = stride
        self.num_workers = num_workers if num_workers else os.cpu_count()

        self.datasets = {}

    # função útil
    def normalize_data(self, data, target_min=-1, target_max=1):
        """Function responsible for normalizing images in the range (-1,1)

        Parameters
        ----------
        data : np.ndarray
            Sample (image), with 3 channels
        target_min : int
            Min value of target to normalize data.
        target_max : int
            Max value of target to normalize data.

        Returns
        -------
        np.ndarray
            Sample (image) normalized.
        """
        data_min, data_max = data.min(), data.max()
        return target_min + (data - data_min) * (target_max - target_min) / (data_max - data_min)
    
    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = [self.normalize_data(image) for image in TiffReader(self.train_path / "train")] # lendo imagens e normalizando
            train_label_reader = PNGReader(self.annotations_path / "train")

            # Aplica o ratio para limitar a quantidade de dados de treinamento
            num_train_samples = int(len(train_img_reader) * self.ratio)
            if num_train_samples < len(train_img_reader):
                indices = random.sample(range(len(train_img_reader)), num_train_samples)
                train_img_reader = [train_img_reader[i] for i in indices]
                train_label_reader = [train_label_reader[i] for i in indices]
            
            # Criar dataset para treinamento
            self.datasets['train'] = SupervisedDatasetPatches(
                readers=[train_img_reader, train_label_reader],
                transforms=self.transforms,
                patch_size=self.patch_size,
                stride=self.stride
            )
            del train_img_reader, train_label_reader
            gc.collect()

            val_img_reader = [self.normalize_data(image) for image in TiffReader(self.train_path / "val")]
            val_label_reader = PNGReader(self.annotations_path / "val")

            self.datasets["val"] = SupervisedDatasetPatches(
                readers=[val_img_reader, val_label_reader],
                transforms=self.transforms,
                patch_size=self.patch_size,
                stride=self.stride
            )
            del val_img_reader, val_label_reader
            gc.collect()
        
        elif stage == "test" or stage == "predict":
            test_img_reader = [self.normalize_data(image) for image in TiffReader(self.train_path / "test")]
            test_label_reader = PNGReader(self.annotations_path / "test")

            test_dataset = SupervisedDatasetPatches(
                readers=[test_img_reader, test_label_reader],
                transforms=self.transforms,
                patch_size=self.patch_size,
                stride=self.stride
            )
            del test_img_reader, test_label_reader
            gc.collect()

            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True, 
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True, 
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True, 
            drop_last=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True, 
            drop_last=False
        )

if __name__ == "__main__":
    print("___START EXPERIMENT___")

    # Inicializa o analisador de argumentos
    parser = argparse.ArgumentParser(description="Script for experiments with SAM v1")
    parser.add_argument('--config', type=str, help="Caminho para o arquivo de configuração JSON", required=True)
    
    args = parser.parse_args()

    # Carregar configurações do JSON
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extrair os parâmetros do JSON
    name = config["name"]
    image_size = config["image_size"]
    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    alpha = config["alpha"]
    rank = config["rank"]
    n = config["n"]
    epochs = config["epochs"]
    data_ratios = config["data_ratios"]
    vit_model = config["vit_model"]
    checkpoint_sam = config["checkpoint_sam"]
    train_path = config["train_path"]
    annotation_path = config["annotation_path"]

    # Verificação do tipo para data_ratios
    if not isinstance(data_ratios, list):
        raise ValueError("O parâmetro 'data_ratios' no JSON precisa ser uma lista.")

    # Exibir parâmetros em formato de tabela
    print(20*'-')
    print("\n--- Parâmetros de Configuração ---")
    print(f"{'Parâmetro':<20} {'Valor'}")
    print("-" * 40)
    print(f"{'name':<20} {name}")
    print(f"{'image_size':<20} {image_size}")
    print(f"{'num_classes':<20} {num_classes}")
    print(f"{'batch_size':<20} {batch_size}")
    print(f"{'alpha':<20} {alpha}")
    print(f"{'rank':<20} {rank}")
    print(f"{'n':<20} {n}")
    print(f"{'epochs':<20} {epochs}")
    print(f"{'data_ratios':<20} {data_ratios}")
    print(f"{'vit_model':<20} {vit_model}")
    print(f"{'checkpoint_sam':<20} {checkpoint_sam}")
    print(f"{'train_path':<20} {train_path}")
    print(f"{'annotation_path':<20} {annotation_path}")
    print(20*'-')

    _init_experiment(
        name=name,
        image_size=image_size, 
        num_classes=num_classes,
        batch_size=batch_size,
        alpha=alpha,
        rank=rank,
        N=n, 
        epochs=epochs, 
        data_ratios=data_ratios,
        vit_model=vit_model,
        checkpoint_path=checkpoint_sam,
        train_path=train_path,
        annotation_path=annotation_path,
        config_path=args.config
    )
    print("___END OF EXPERIMENT___")
    print("Good Night ;p")