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

from scipy.ndimage import gaussian_gradient_magnitude, laplace

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from minerva.data.datasets.supervised_dataset import SupervisedReconstructionDataset
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.models.nets.image.segment_anything.sam_lora import SAMLoRA
from minerva.transforms.transform import _Transform
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline

import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

def log_memory():
    print(f"CPU Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

def _init_experiment(
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
        log_memory()
        
        for _ in tqdm(range(N)):
            model = SAMLoRA(
                image_size=image_size,
                num_classes=num_classes,
                alpha=alpha,
                rank=rank,
                checkpoint=checkpoint_path
            )
            
            train_metrics, val_metrics, test_metrics = train_and_evaluate(image_size, model, ratio, epochs, batch_size=batch_size, train_path=train_path, annotation_path=annotation_path)
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
    plt.bar(["1% dos Dados", "100% dos Dados"], [means[0.01], means[1.0]], yerr=[stds[0.01], stds[1.0]], capsize=5)
    plt.ylabel('mIoU Médio')
    plt.title('Comparação de mIoU para 1% e 100% dos Dados')
    plt.savefig(save_dir / 'miou_comparison.png')
    # plt.close()
    
    def extract_metric(metrics, metric_name):
        # Extrai o valor de cada época para uma métrica específica
        return [[epoch_metrics[metric_name] for epoch_metrics in epoch] for epoch in metrics]

    for ratio, label in zip(data_ratios, ["1% dos Dados", "100% dos Dados"]):
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
        image_size,
        model, 
        ratio, 
        epochs, 
        train_path,
        annotation_path,
        batch_size=8,
        ):

    # Criando data module com o ratio definido
    data_module = PatchingTolstayaModule(
        train_path=train_path,
        annotations_path=annotation_path,
        ratio=ratio,
        patch_size=image_size,
        stride=32,
        batch_size=batch_size
    )

    # current_date = datetime.now().strftime("%Y-%m-%d")

    # Define o callback para salvar o modelo com base no menor valor da métrica de validação
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss", # Métrica para monitorar
    #     dirpath="./checkpoints", # Diretório onde os checkpoints serão salvos
    #     filename=f"sam_model-final_train-raio-{ratio}-{current_date}-{{epoch:02d}}-{{val_loss:.2f}}", # Nome do arquivo do checkpoint
    #     save_top_k=1, # Quantos melhores checkpoints salvar (no caso, o melhor)
    #     mode="min", # Como a métrica deve ser tratada (no caso, 'min' significa que menor valor de val_loss é melhor)
    # )
    
    logger = CSVLogger("logs", name="sam_model")

    # Inicialize o logger e o treinador para cada execução
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        # callbacks=[checkpoint_callback],
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

class PatchingTolstayaModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        ratio: float = 1.0,
        patch_size: int = 255,
        stride: int = 32,
        batch_size: int = 8,
        transforms: _Transform = None,
        num_workers: int = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.ratio = ratio
        self.transforms = transforms
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.num_workers = num_workers if num_workers else os.cpu_count()

        self.datasets = {}

    # função útil
    def normalize_data(self, data, target_min=-1, target_max=1):
        """
        Função responsável por normalizar as imagens no intervalo (-1,1)
        """
        data_min, data_max = data.min(), data.max()
        return target_min + (data - data_min) * (target_max - target_min) / (data_max - data_min)

    # função útil
    def generate_depth_channel(self, shape):
        """
        Função responsável por criar o canal de profundidade, com 0 no topo e 1 na base.
        """
        depth_channel = np.linspace(0, 1, shape[0]).reshape(-1, 1)
        return np.tile(depth_channel, (1, shape[1]))
    
    # função útil
    def extract_patches(self, data, patch_size=255, stride=32, img_type='image'):
        patches = []
        if img_type == 'image': # caso seja imagens de entrada (h, w, c)
            h, w, _ = data.shape
        else: # caso seja labels de entrada (h, w)
            h, w = data.shape
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = data[i:i + patch_size, j:j + patch_size]
                if img_type == 'image':
                    patches.append(patch.transpose(2, 0, 1).astype(np.float32)) # o SAM só recebe (C H W)
                else:
                    patches.append(patch.astype(np.int64))
        return np.array(patches)
    
    # funcao utils
    def generate_facies_probability_channel(self, patch, facies_probabilities):
        prob_map = np.random.choice(
            len(facies_probabilities), size=patch.shape, p=facies_probabilities
            )
        return prob_map

    # funcao util
    def generate_amplitude_gradient_channel(self, patch):
        # Usa gradiente de amplitude com suavização
        gradient_channel = gaussian_gradient_magnitude(patch, sigma=1)
        return gradient_channel / np.max(gradient_channel)  # Normalização

    # funcao util
    def generate_curvature_channel(self, patch):
        # Calcula a curvatura usando o filtro de Laplace
        curvature_channel = laplace(patch)
        return curvature_channel / np.max(np.abs(curvature_channel))  # Normaliza para [-1, 1]

    # função útil
    def generate_image_with_depth(self, normalized_images):
        two_channel_images = []
        for img in normalized_images:  # Garantir que está trabalhando com imagens normalizadas
            depth_channel = self.generate_depth_channel(img.shape[:2])  # Gerar canal de profundidade para esta imagem

            if img.shape[-1] > 1:  # Se a imagem tiver múltiplos canais
                # Escolher apenas o primeiro canal (exemplo, pode ser qualquer canal)
                img = img[:, :, 0:1]  # Seleciona o primeiro canal e mantém as dimensões (H, W, 1)
            
            # gerando canal 3 (teste)
            # facies_probabilities = np.array([0.2857, 0.1207, 0.4696, 0.0747, 0.0361, 0.0132])
            # canal_3 = self.generate_facies_probability_channel(img, facies_probabilities)
            # canal_3 = self.generate_amplitude_gradient_channel(img)
            # canal_3 = self.generate_curvature_channel(img)

            # Concatenar o primeiro canal com o depth_channel
            depth_channel = np.expand_dims(depth_channel, axis=-1)  # Tornar (H, W, 1)
            two_channel_image = np.concatenate((img, depth_channel), axis=-1)  # Concatenar ao longo do eixo dos canais
            two_channel_images.append(two_channel_image)  # Adicionar ao array final
        return two_channel_images
    
    # função util
    def horizontal_flip(self, image, label):
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Experado image com type <class 'numpy.ndarray'>, mas foi recebido {type(image)}")
        if not isinstance(label, np.ndarray):
            raise ValueError(f"Experado label com type <class 'numpy.ndarray'>, mas foi recebido {type(label)}")

        if image.shape[0] != 3 or image.shape[1] != self.patch_size or image.shape[2] != self.patch_size:
            raise ValueError(f"Experado image com shape (C=3 H=patch_size W=patch_size), mas foi recebido {image.shape}")
        if label.shape[0] != self.patch_size or label.shape[1] != self.patch_size:
            raise ValueError(f"Experado label com shape (H=patch_size W=patch_size), mas foi recebido {label.shape}")

        image_flipped = np.flip(image, axis=2)
        label_flipped = np.flip(label, axis=1)

        # if any(s < 0 for s in image_flipped.strides):
        #     print(f"array com strides negativos detectado")

        return image_flipped.copy(), label_flipped.copy()
    
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
            
            # Gerar imagens com canais de profundidade
            # train_img_reader = self.generate_image_with_depth(train_img_reader)
            
            # Gerar patches em batches
            patches_img_generator = []
            for image in train_img_reader:
                patches_img_generator.extend(self.extract_patches(image, self.patch_size, self.stride))
            patches_label_generator = []
            for image in train_label_reader:
                patches_label_generator.extend(self.extract_patches(image, self.patch_size, self.stride, img_type='label'))

            # """ augmentation """
            # # Aplicar augmentações nas imagens e labels, somente para as amostras com 3, 4 ou 6 classes
            # augmented_img_generator = []
            # augmented_label_generator = []
            # for img, label in zip(patches_img_generator, patches_label_generator):
            #     # TODO tá estourando nessa parte
            #     augmented_img_generator.append(img)
            #     augmented_label_generator.append(label)
            #     # Verificar o número de classes na amostra
            #     unique_classes = np.unique(label)
            #     num_classes = len(unique_classes)
                
            #     if num_classes in [3, 4, 6]:  # Apenas aplicar augmentação nas amostras com 3, 4 ou 6 classes
            #         img_horizontal_flip, label_horizontal_flip = self.horizontal_flip(img, label)
            #         augmented_img_generator.append(img_horizontal_flip)
            #         augmented_label_generator.append(label_horizontal_flip)
            
            # Criar dataset para treinamento
            self.datasets["train"] = SupervisedReconstructionDataset(
                readers=[patches_img_generator, patches_label_generator],
                transforms=self.transforms,
            )
            del train_img_reader, train_label_reader
            del patches_img_generator, patches_label_generator
            # del augmented_img_generator, augmented_label_generator
            gc.collect()

            val_img_reader = [self.normalize_data(image) for image in TiffReader(self.train_path / "val")]
            val_label_reader = PNGReader(self.annotations_path / "val")

            # gerar imagens com canais de profundidade
            # val_img_reader = self.generate_image_with_depth(val_img_reader)

            # Gerar patches em batches
            patches_img_generator = []
            for image in val_img_reader:
                patches_img_generator.extend(self.extract_patches(image, self.patch_size, self.stride))
            patches_label_generator = []
            for image in val_label_reader:
                patches_label_generator.extend(self.extract_patches(image, self.patch_size, self.stride, img_type='label'))

            self.datasets["val"] = SupervisedReconstructionDataset(
                readers=[patches_img_generator, patches_label_generator],
                transforms=self.transforms,
            )
            del val_img_reader, val_label_reader
            del patches_img_generator, patches_label_generator
            gc.collect()
        
        elif stage == "test" or stage == "predict":
            test_img_reader = [self.normalize_data(image) for image in TiffReader(self.train_path / "test")]
            test_label_reader = PNGReader(self.annotations_path / "test")

            # gerar imagens com canais de profundidade
            # test_img_reader = self.generate_image_with_depth(test_img_reader)

            # Gerar patches em batches
            patches_img_generator = []
            for image in test_img_reader:
                patches_img_generator.extend(self.extract_patches(image, self.patch_size, self.stride))
            patches_label_generator = []
            for image in test_label_reader:
                patches_label_generator.extend(self.extract_patches(image, self.patch_size, self.stride, img_type='label'))

            test_dataset = SupervisedReconstructionDataset(
                readers=[patches_img_generator, patches_label_generator],
                transforms=self.transforms,
            )
            del test_img_reader, test_label_reader
            del patches_img_generator, patches_label_generator
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

    def worker_init_fn(self, worker_id):
        random.seed(18 + worker_id)

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
        checkpoint_path=checkpoint_sam,
        train_path=train_path,
        annotation_path=annotation_path,
        config_path=args.config
    )
    print("___END OF EXPERIMENT___")
    print("Good Night ;p")