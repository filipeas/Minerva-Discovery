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

def _init_experiment(
        image_size=512,
        num_classes=5,
        batch_size=2,
        alpha=1,
        rank=4,
        N=10, 
        epochs=500, 
        data_ratios=[0.01, 1.0],
        checkpoint_path="/workspaces/Minerva-Dev-Container/shared_data/weights_sam/checkpoints_sam/sam_vit_b_01ec64.pth",
        train_path="/workspaces/Minerva-Dev-Containe",
        annotation_path="/workspaces/Minerva-Dev-Containe"
        ):
    
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
    plt.savefig('miou_comparison.png')
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
        plt.savefig(f'train_val_loss_{int(ratio*100)}.png')
        
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
        plt.savefig(f'train_miou_{int(ratio*100)}.png')
        # plt.close()

# Função para treinar e testar o modelo
def train_and_evaluate(
        image_size,
        model, 
        ratio, 
        epochs, 
        batch_size=4,
        train_path="/workspaces/Minerva-Dev-Container/shared_data/seismic/f3_segmentation/images",
        annotation_path="/workspaces/Minerva-Dev-Container/shared_data/seismic/f3_segmentation/annotations"
        ):
    transformImage = RandomGeneratorForImage((image_size, image_size))
    transformLabel = RandomGeneratorForLabel((image_size, image_size))

    # Criando data module com o ratio definido
    data_module = F3DataModule(
        train_path=train_path,
        annotations_path=annotation_path,
        transforms=[transformImage, transformLabel],
        batch_size=batch_size,
        ratio=ratio
    )

    # Inicialize o logger e o treinador para cada execução
    # logger = TensorBoardLogger("logs", name=f"sam_model_{int(ratio*100)}")
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

    del train_path, annotation_path, transformImage, transformLabel, data_module, trainer, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    return train_metrics, val_metrics, test_metrics

class RandomGeneratorForImage(_Transform):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        channels, x, y = sample.shape
        
        # Redimensiona cada canal individualmente para evitar o erro de dimensão
        resized_channels = [
            zoom(sample[c], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            for c in range(channels)
        ]
        
        # Converte a lista de canais redimensionados de volta para um array numpy e, em seguida, para um tensor PyTorch
        image = np.stack(resized_channels, axis=0).astype(np.float32)
        image = torch.from_numpy(image)
        return image

class RandomGeneratorForLabel(_Transform):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        label = sample

        x, y = label.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = torch.from_numpy(label.astype(np.float32))
        return label.long()

def rotate_and_flip_patch(patch_img, patch_label):
    rotate = cv2.ROTATE_90_CLOCKWISE
    # Rotaciona -90 graus (equivalente a transpor e depois inverter verticalmente)
    patch_img_rotated = cv2.rotate(patch_img, rotate)
    patch_label_rotated = cv2.rotate(patch_label, rotate)
    # Espelha horizontalmente
    patch_img_flipped = cv2.flip(patch_img_rotated, 1)
    patch_label_flipped = cv2.flip(patch_label_rotated, 1)
    return patch_img_flipped, patch_label_flipped

def resize_image(image, target_shape):
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)

def cria_patchs_sismicas(images, labels, patch_size=(512,512), step=512):
    all_img_patches = []
    all_label_patches = []
    for i, large_image in enumerate(images): # iterando as imagens
        target_shape = ((large_image.shape[1] // patch_size[1] + 1) * patch_size[1],
                        (large_image.shape[0] // patch_size[0] + 1) * patch_size[0])
        
        large_image_resized = resize_image(np.array(large_image), target_shape)
        label = np.array(labels[i]).astype(np.uint8)
        large_label_resized = resize_image(label, target_shape)
        
        # Verifica se as dimensões redimensionadas são múltiplos exatos do patch_size
        assert large_image_resized.shape[0] % patch_size[0] == 0, "Altura da imagem não é múltipla do patch_size."
        assert large_image_resized.shape[1] % patch_size[1] == 0, "Largura da imagem não é múltipla do patch_size."
        assert large_label_resized.shape[0] % patch_size[0] == 0, "Altura da label não é múltipla do patch_size."
        assert large_label_resized.shape[1] % patch_size[1] == 0, "Largura da label não é múltipla do patch_size."

        patches_img = patchify(large_image_resized, patch_size=patch_size + (3,), step=step)
        patches_label = patchify(large_label_resized, patch_size=patch_size, step=step)
        
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, 0]  # Remove a dimensão extra da Imagem
                single_patch_label = patches_label[i, j]  # Remove a dimensão extra da Label
                
                # Converte o patch da imagem para float32
                single_patch_img = np.transpose(single_patch_img, (2, 0, 1)).astype(np.float32)
                # Converte o patch do rótulo para int64 (long)
                single_patch_label = single_patch_label.astype(np.int64)
                
                all_img_patches.append(single_patch_img)
                all_label_patches.append(single_patch_label)
    return np.array(all_img_patches), np.array(all_label_patches)

class F3DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        transforms: _Transform = None,
        batch_size: int = 1,
        num_workers: int = None,
        ratio: float = 1.0
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.ratio = ratio

        self.datasets = {}

    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = TiffReader(self.train_path / "train")
            train_label_reader = PNGReader(self.annotations_path / "train")

            # Aplica o ratio para limitar a quantidade de dados de treinamento
            num_train_samples = int(len(train_img_reader) * self.ratio)
            if num_train_samples < len(train_img_reader):
                indices = random.sample(range(len(train_img_reader)), num_train_samples)
                train_img_reader = [train_img_reader[i] for i in indices]
                train_label_reader = [train_label_reader[i] for i in indices]
            
            train_imgs, train_labels = cria_patchs_sismicas(train_img_reader, train_label_reader)

            del train_img_reader, train_label_reader
            gc.collect()
            torch.cuda.empty_cache()

            train_dataset = SupervisedReconstructionDataset(
                readers=[train_imgs, train_labels],
                transforms=self.transforms,
            )
            print(f"Qtd. train_dataset: {len(train_dataset)}")

            val_img_reader = TiffReader(self.train_path / "val")
            val_label_reader = PNGReader(self.annotations_path / "val")

            val_imgs, val_labels = cria_patchs_sismicas(val_img_reader, val_label_reader)

            val_dataset = SupervisedReconstructionDataset(
                readers=[val_imgs, val_labels],
                transforms=self.transforms,
            )
            print(f"Qtd. val_dataset: {len(val_dataset)}")

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

            del train_imgs, train_labels
            del val_img_reader, val_label_reader, val_imgs, val_labels
            gc.collect()
            torch.cuda.empty_cache()

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReader(self.train_path / "test")
            test_label_reader = PNGReader(self.annotations_path / "test")

            test_imgs, test_labels = cria_patchs_sismicas(test_img_reader, test_label_reader)

            test_dataset = SupervisedReconstructionDataset(
                readers=[test_imgs, test_labels],
                transforms=self.transforms,
            )
            print(f"Qtd test_dataset: {len(test_dataset)}")
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

            del test_img_reader, test_label_reader, test_imgs, test_labels
            gc.collect()
            torch.cuda.empty_cache()
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
        annotation_path=annotation_path
    )
    print("___END OF EXPERIMENT___")
    print("Good Night ;p")