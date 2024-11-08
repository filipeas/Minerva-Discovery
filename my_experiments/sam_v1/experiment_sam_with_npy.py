import lightning as L
import numpy as np
import gc
import time
import torch
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
from einops import repeat
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import shutil

def _init_experiment(
        config_path,
        image_size=512,
        num_classes=5,
        batch_size=2,
        alpha=1,
        rank=4,
        N=10, 
        epochs=500, 
        data_ratios=[0.01, 1.0],
        checkpoint_path="/workspaces/Minerva-Dev-Container/shared_data/weights_sam/checkpoints_sam/sam_vit_b_01ec64.pth",
        train_seismic="",
        train_labels="",
        test_seismic="",
        test_labels="",
        ):
    # Gerar um identificador único baseado no timestamp atual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Criar o diretório único para salvar os gráficos
    save_dir = Path(os.getcwd()) / f"experiment_{timestamp}"
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
                num_classes=num_classes, # considera 6 pois internamente o sam faz +1 pro background
                # pixel_mean=pixel_mean,
                # # pixel_std=pixel_std,
                alpha=alpha,
                rank=rank,
                # apply_lora_vision_encoder=apply_lora_vision_encoder,
                # # apply_lora_mask_decoder=apply_lora_mask_decoder,
                # # frozen_vision_encoder=frozen_vision_encoder,
                # # frozen_prompt_encoder=frozen_prompt_encoder,
                # # frozen_mask_decoder=frozen_mask_decoder,
                # # vit_model=vit_model,
                checkpoint=checkpoint_path,
                # train_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
                # val_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
                # test_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=6)}
            )
            
            train_metrics, val_metrics, test_metrics = train_and_evaluate(image_size, model, ratio, epochs, batch_size=batch_size, train_seismic=train_seismic, train_labels=train_labels, test_seismic=test_seismic, test_labels=test_labels)
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
        batch_size=4,
        train_seismic="",
        train_labels="",
        test_seismic="",
        test_labels=""
        ):
    transformImage = RandomGeneratorForImage((image_size, image_size))
    transformLabel = RandomGeneratorForLabel((image_size, image_size))

    # Criando data module com o ratio definido
    data_module = F3DataModule(
        train_seismic=np.load(os.path.join(train_seismic)),
        train_labels=np.load(os.path.join(train_labels)),
        test_seismic=np.load(os.path.join(test_seismic)),
        test_labels=np.load(os.path.join(test_labels)),
        patch_size=image_size,
        step=image_size,
        transforms=[transformImage, transformLabel],
        batch_size=batch_size,
        ratio=ratio
    )

    # Inicialize o logger e o treinador para cada execução
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger
    )

    # Inicialize o pipeline e execute o treinamento
    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        save_run_status=True
    )
    pipeline.run(data=data_module, task="fit")
    test_metrics = pipeline.run(data=data_module, task="test")

    # Extrair métricas (exemplo: mIoU e perda)
    train_metrics = pipeline.model.train_metrics_acc # todas as métricas de treino
    pipeline.model.train_metrics_acc = []
    val_metrics = pipeline.model.val_metrics_acc[1:] # todas as métricas de validacao
    pipeline.model.val_metrics_acc = []
    
    del transformImage, transformLabel, data_module, trainer, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    return train_metrics, val_metrics, test_metrics

class npyReader():
    def __init__(self, train_seismic, train_labels, patch_size=512, step=512):
        self.images = np.concatenate((
            self._cria_patchs_sismicas_inline(train_seismic, patch_size=(patch_size, patch_size), step=step), 
            self._cria_patchs_sismicas_crossline(train_seismic, patch_size=(patch_size, patch_size), step=step)
        ), axis=0)
        
        self.masks = np.concatenate((
            self._cria_patchs_labels_inline(train_labels, patch_size=(patch_size, patch_size), step=step), 
            self._cria_patchs_labels_crossline(train_labels, patch_size=(patch_size, patch_size), step=step)
        ), axis=0)

    def _rotate_and_flip_patch(self, patch):
        # Rotaciona -90 graus (equivalente a transpor e depois inverter verticalmente)
        patch_rotated = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        # Espelha horizontalmente
        patch_flipped = cv2.flip(patch_rotated, 1)
        return patch_flipped
    
    def _resize_image(self, image, target_shape):
        return cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)

    def _cria_patchs_sismicas_crossline(self, images, patch_size=(256,256), step=256):
        all_img_patches = []
        for crossline_idx in range(images.shape[1]): # iterando no crossline
            large_image = images[:, crossline_idx, :]
            target_shape = ((large_image.shape[1] // patch_size[1] + 1) * patch_size[1],
                            (large_image.shape[0] // patch_size[0] + 1) * patch_size[0])
            large_image_resized = self._resize_image(large_image, target_shape)

            patches_img = patchify(large_image_resized, patch_size=patch_size, step=step)  #Step=256 for 256 patches means no overlap

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = self._rotate_and_flip_patch(single_patch_img) # Aplica a rotação e espelhamento
                    all_img_patches.append(single_patch_img)
        return np.array(all_img_patches)

    def _cria_patchs_sismicas_inline(self, images, patch_size=(256,256), step=256):
        all_img_patches = []
        for inline_idx in range(images.shape[0]): # iterando no inline
            large_image = images[inline_idx, :, :]
            target_shape = ((large_image.shape[1] // patch_size[1] + 1) * patch_size[1],
                            (large_image.shape[0] // patch_size[0] + 1) * patch_size[0])
            large_image_resized = self._resize_image(large_image, target_shape)

            patches_img = patchify(large_image_resized, patch_size=patch_size, step=step)  #Step=256 for 256 patches means no overlap

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = self._rotate_and_flip_patch(single_patch_img) # Aplica a rotação e espelhamento
                    all_img_patches.append(single_patch_img)
        return np.array(all_img_patches)

    def _cria_patchs_labels_crossline(self, images_labels, patch_size=(256,256), step=256):
        all_mask_patches = []
        for crossline_idx in range(images_labels.shape[1]): # iterando no crossline
            large_mask = images_labels[:, crossline_idx, :]
            target_shape = ((large_mask.shape[1] // patch_size[1] + 1) * patch_size[1],
                            (large_mask.shape[0] // patch_size[0] + 1) * patch_size[0])
            large_mask_resized = self._resize_image(large_mask, target_shape)

            patches_mask = patchify(large_mask_resized, patch_size=patch_size, step=step)  #Step=256 for 256 patches means no overlap

            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i,j,:,:]
                    single_patch_mask = self._rotate_and_flip_patch(single_patch_mask) # Aplica a rotação e espelhamento
                    all_mask_patches.append(single_patch_mask)
        return np.array(all_mask_patches)

    def _cria_patchs_labels_inline(self, images_labels, patch_size=(256,256), step=256):
        all_mask_patches = []
        for inline_idx in range(images_labels.shape[0]): # iterando no inline
            large_mask = images_labels[inline_idx, :, :]
            target_shape = ((large_mask.shape[1] // patch_size[1] + 1) * patch_size[1],
                            (large_mask.shape[0] // patch_size[0] + 1) * patch_size[0])
            large_mask_resized = self._resize_image(large_mask, target_shape)

            patches_mask = patchify(large_mask_resized, patch_size=patch_size, step=step)  #Step=256 for 256 patches means no overlap

            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i,j,:,:]
                    single_patch_mask = self._rotate_and_flip_patch(single_patch_mask) # Aplica a rotação e espelhamento
                    all_mask_patches.append(single_patch_mask)
        return np.array(all_mask_patches)

class RandomGeneratorForImage(_Transform):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        
        return image

class RandomGeneratorForLabel(_Transform):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, label: np.ndarray) -> np.ndarray:
        x, y = label.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = torch.from_numpy(label.astype(np.float32))
        return label.long()

class F3DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_seismic,
        train_labels,
        test_seismic,
        test_labels,
        patch_size: int = 512,
        step: int = 512,
        transforms: _Transform = None,
        batch_size: int = 1,
        num_workers: int = None,
        ratio: float = 1.0
    ):
        super().__init__()
        self.train_seismic = train_seismic
        self.train_labels = train_labels
        self.test_seismic = test_seismic
        self.test_labels = test_labels
        self.patch_size = patch_size
        self.step = step
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.ratio = ratio

        self.datasets = {}

    def setup(self, stage=None):
        if stage == "fit":
            npy_files = npyReader(
                train_seismic=self.train_seismic, 
                train_labels=self.train_labels, 
                patch_size=self.patch_size, 
                step=self.step
            )
            train_imgs = npy_files.images
            train_labels = npy_files.masks

            # Dividir 80/20 para treino e validação
            num_train_samples = int(len(train_imgs) * 0.8)
            indices = list(range(len(train_imgs)))
            random.shuffle(indices)
            train_indices = indices[:num_train_samples]
            val_indices = indices[num_train_samples:]

            train_img_reader = [train_imgs[i] for i in train_indices]
            train_label_reader = [train_labels[i] for i in train_indices]
            val_img_reader = [train_imgs[i] for i in val_indices]
            val_label_reader = [train_labels[i] for i in val_indices]

            # Aplica o ratio para limitar a quantidade de dados de treinamento
            num_train_samples = int(len(train_img_reader) * self.ratio)
            if num_train_samples < len(train_img_reader):
                indices = random.sample(range(len(train_img_reader)), num_train_samples)
                train_img_reader = [train_img_reader[i] for i in indices]
                train_label_reader = [train_label_reader[i] for i in indices]
            
            train_dataset = SupervisedReconstructionDataset(
                readers=[train_img_reader, train_label_reader],
                transforms=self.transforms,
            )
            print(f"Qtd. train_dataset: {len(train_dataset)}")

            val_dataset = SupervisedReconstructionDataset(
                readers=[val_img_reader, val_label_reader],
                transforms=self.transforms,
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            npy_files = npyReader(
                train_seismic=self.test_seismic, 
                train_labels=self.test_labels, 
                patch_size=self.patch_size, 
                step=self.step
            )
            test_imgs = npy_files.images
            test_labels = npy_files.masks

            test_dataset = SupervisedReconstructionDataset(
                readers=[test_imgs, test_labels],
                transforms=self.transforms,
            )
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True, 
            worker_init_fn=self.worker_init_fn,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True, 
            worker_init_fn=self.worker_init_fn,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True, 
            worker_init_fn=self.worker_init_fn,
            drop_last=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True, 
            worker_init_fn=self.worker_init_fn,
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
    image_size = config["image_size"]
    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    alpha = config["alpha"]
    rank = config["rank"]
    n = config["n"]
    epochs = config["epochs"]
    data_ratios = config["data_ratios"]
    checkpoint_sam = config["checkpoint_sam"]
    train_seismic = config["train_seismic"]
    train_labels = config["train_labels"]
    test_seismic = config["test_seismic"]
    test_labels = config["test_labels"]

    # Verificação do tipo para data_ratios
    if not isinstance(data_ratios, list):
        raise ValueError("O parâmetro 'data_ratios' no JSON precisa ser uma lista.")

    # Exibir parâmetros em formato de tabela
    print(20*'-')
    print("\n--- Parâmetros de Configuração ---")
    print(f"{'Parâmetro':<20} {'Valor'}")
    print("-" * 40)
    print(f"{'image_size':<20} {image_size}")
    print(f"{'num_classes':<20} {num_classes}")
    print(f"{'batch_size':<20} {batch_size}")
    print(f"{'alpha':<20} {alpha}")
    print(f"{'rank':<20} {rank}")
    print(f"{'n':<20} {n}")
    print(f"{'epochs':<20} {epochs}")
    print(f"{'data_ratios':<20} {data_ratios}")
    print(f"{'checkpoint_sam':<20} {checkpoint_sam}")
    print(f"{'train_seismic':<20} {train_seismic}")
    print(f"{'train_labels':<20} {train_labels}")
    print(f"{'test_seismic':<20} {test_seismic}")
    print(f"{'test_labels':<20} {test_labels}")
    print(20*'-')

    _init_experiment(
        config_path=args.config,
        image_size=image_size, 
        num_classes=num_classes,
        batch_size=batch_size,
        alpha=alpha,
        rank=rank,
        N=n, 
        epochs=epochs, 
        data_ratios=data_ratios,
        checkpoint_path=checkpoint_sam,
        train_seismic=train_seismic,
        train_labels=train_labels,
        test_seismic=test_seismic,
        test_labels=test_labels,
    )
    print("___END OF EXPERIMENT___")
    print("Good Night ;p")