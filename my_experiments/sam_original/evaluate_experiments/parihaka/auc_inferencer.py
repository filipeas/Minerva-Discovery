import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import lightning as L
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import find_contours
from torchmetrics import JaccardIndex
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from grad_cam import GradCAM
from typing import Optional
import torchvision.utils as vutils

def plot_all(
        image, 
        label, 
        pred=None, 
        diff=None, 
        score=None, 
        point_coords=None, 
        point_labels=None, 
        cam=None, 
        i=0, 
        batch_idx=0, 
        process_func='', 
        sugest_filename="sample_pred_all", 
        model_idx="model_name", 
        using_methodology=1
):
    """
    Plota as imagens lado a lado: imagem original, label, predição (opcional), diff (opcional) e Grad-CAM (opcional).
    Pontos acumulados são exibidos sobre as imagens.
    Uma colorbar com os valores mínimo e máximo do Grad-CAM é adicionada à direita do subplot do Grad-CAM.
    """
    # Determina o número de subplots com base nos argumentos fornecidos
    num_subplots = 2  # Imagem original e label são obrigatórios
    if pred is not None:
        num_subplots += 1
    if diff is not None:
        num_subplots += 1
    if cam is not None:
        num_subplots += 1

    plt.clf()
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

    # Função para normalizar os dados
    def normalize(data):
        if data.dtype == np.float32 or data.dtype == np.float64:
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        elif data.dtype == np.uint8:
            return np.clip(data, 0, 255)
        return data

    image = normalize(image)
    label = normalize(label)
    if pred is not None:
        pred = normalize(pred)
    if diff is not None:
        diff = normalize(diff)

    # Plot 1: Imagem original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Plot 2: Label
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title("Label")
    axes[1].axis('off')

    # Plot 3: Predição acumulada (opcional)
    next_idx = 2
    if pred is not None:
        axes[next_idx].imshow(pred, cmap='gray')
        axes[next_idx].set_title(f"Pred - Score: {score}" if score is not None else "Pred")
        axes[next_idx].axis('off')
        next_idx += 1

    # Plot 4: Diferença entre label e predição (opcional)
    if diff is not None:
        axes[next_idx].imshow(diff, cmap='gray')
        axes[next_idx].set_title("Difference (Label - Pred)")
        axes[next_idx].axis('off')
        next_idx += 1

    # Plot 5: Grad-CAM (opcional) com coluna de colorbar
    if cam is not None:
        im = axes[next_idx].imshow(cam, cmap='jet')
        axes[next_idx].set_title("Grad-CAM sob a camada 'neck'")
        axes[next_idx].axis('off')
        # Adiciona uma colorbar à direita do subplot do Grad-CAM
        cbar = fig.colorbar(im, ax=axes[next_idx], fraction=0.046, pad=0.04)
        # Define ticks com os valores mínimo e máximo do cam
        cbar.set_ticks([np.min(cam), np.max(cam)])
        cbar.set_ticklabels(["Min: {:.2f}".format(np.min(cam)), "Max: {:.2f}".format(np.max(cam))])

    # Adiciona os pontos em todas as imagens (opcional)
    if point_coords is not None and point_labels is not None:
        for ax in axes:
            for (x, y), lbl in zip(point_coords, point_labels):
                color = 'green' if lbl == 1 else 'red'
                ax.scatter(x, y, color=color, s=50, edgecolors='white')

    plt.tight_layout()
    os.makedirs(f"tmp/debug_region/methodology_{using_methodology}/{model_idx}/{process_func}/{batch_idx}", exist_ok=True)
    filename = f"{sugest_filename}_{i}"  # Nome do arquivo
    output_path = f"tmp/debug_region/methodology_{using_methodology}/{model_idx}/{process_func}/{batch_idx}/{filename}.png"
    plt.savefig(output_path)
    plt.close()

def plot_facie_with_prompts(
        facie_idx, 
        point_coords_positive, 
        point_coords_negative, 
        region, 
        batch_idx, 
        process_func, 
        model_idx="model_name", 
        using_methodology=1
):
    # Verificar se a imagem 'region' está em escala de cinza (preto e branco)
    if region.ndim != 2:
        raise ValueError("A imagem 'region' deve ser uma imagem em preto e branco (escala de cinza).")

    # Extrair as coordenadas (x, y) da lista de tuplas
    coords_positive = np.array([(x, y) for x, y in point_coords_positive])
    coords_negative = np.array([(x, y) for x, y in point_coords_negative])

    # Criar a figura e o eixo para o plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plotar a imagem 'region' em escala de cinza
    ax.imshow(region, cmap='gray', vmin=0, vmax=255)
    ax.set_title('Imagem com Coordenadas')
    ax.axis('off')

    # Plotar as coordenadas sobre a imagem
    ax.scatter(coords_positive[:, 0], coords_positive[:, 1], c='green', marker='o', s=25)
    ax.scatter(coords_negative[:, 0], coords_negative[:, 1], c='red', marker='o', s=25)

    # Salvar a imagem
    os.makedirs(f"tmp/debug_region/methodology_{using_methodology}/{model_idx}/{process_func}/{batch_idx}", exist_ok=True)
    output_path = f"tmp/debug_region/methodology_{using_methodology}/{model_idx}/{process_func}/{batch_idx}/image_facie_{facie_idx}_with_coordinates.png"
    plt.savefig(output_path)
    plt.close()

def plot_result_batch_preds(
        batch_preds, 
        batch_idx, 
        process_func, 
        model_idx="model_name",
        using_methodology=1
):
    label_cmap = ListedColormap(
        [
            [0.29411764705882354, 0.4392156862745098, 0.7333333333333333],
            [0.5882352941176471, 0.7607843137254902, 0.8666666666666667],
            [0.8901960784313725, 0.9647058823529412, 0.9764705882352941],
            [0.9803921568627451, 0.8745098039215686, 0.4666666666666667],
            [0.9607843137254902, 0.47058823529411764, 0.29411764705882354],
            [0.8470588235294118, 0.1568627450980392, 0.1411764705882353],
            [1.0, 0.7529411764705882, 0.796078431372549], #background
        ]
    )
    
    for sample_idx, sample_preds in enumerate(batch_preds):
        # sample_preds tem formato [num_prompts, H, W]
        for prompt_idx in range(sample_preds.shape[0]):
            # print(f"::DEBUG - Saving image {sample_idx} (Prompt {prompt_idx})")
            # Converter a predição para um array numpy
            image_array = sample_preds[prompt_idx].cpu().numpy()

            # Atualize a verificação para permitir os valores 0 a 5 e o 7 (fundo)
            unique_values = np.unique(image_array)
            if not np.all(np.isin(unique_values, [0, 1, 2, 3, 4, 5, 7])):
                raise ValueError("A imagem contém valores fora do intervalo permitido (0 a 5 ou 7).")

            # print("shape do sample_preds: ", sample_preds[prompt_idx].shape)
            # print("qtd de pixels únicos: ", unique_values)

            # Número de subplots necessários (imagem original + máscaras para cada valor único)
            num_plots = len(unique_values) + 1
            fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))

            # Plotar a imagem original com o mapeamento de cores customizado
            # Usamos vmin=0 e vmax=7 para garantir que o valor 7 seja mapeado para a última cor
            axes[0].imshow(image_array, cmap=label_cmap, vmin=0, vmax=7)
            axes[0].set_title('Imagem')
            axes[0].axis('off')

            # Plotar cada nível (valor) em uma máscara binária
            for i, value in enumerate(unique_values, start=1):
                mask = (image_array == value)
                pixel_count = np.sum(mask)
                binary_image = np.where(mask, 255, 0)

                axes[i].imshow(binary_image, cmap='gray', vmin=0, vmax=255)
                axes[i].set_title(f'Valor {value}\nPixels: {pixel_count}')
                axes[i].axis('off')

            plt.suptitle(f'Sample {sample_idx} - Prompt {prompt_idx}')

            os.makedirs(f"tmp/debug_region/methodology_{using_methodology}/{model_idx}/{process_func}/{batch_idx}", exist_ok=True)
            output_path = f"tmp/debug_region/methodology_{using_methodology}/{model_idx}/{process_func}/{batch_idx}/sample_{sample_idx}_prompt_{prompt_idx}.png"
            plt.savefig(output_path)
            plt.close()

def save_grayscale_image(image_array, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, image_array, cmap="gray")

class AUCInferencer(L.LightningModule):
    def __init__(
            self, 
            model:L.LightningModule, 
            data_module:L.LightningDataModule,
            data_module_setup:str = 'predict',
            multimask_output: bool = False, 
            num_points: int = 3,
            using_methodology: int = 2,
            model_idx: str = "model_name",
            execute_only_predictions: bool = False,
            evaluate_this_samples: Optional[list] = None
            ):
        super().__init__()
        self.model = model
        self.num_points = num_points
        self.multimask_output = multimask_output
        self.set_points()  # Reset accumulated points
        self.miou_metric = JaccardIndex(task="multiclass", num_classes=2).to(self.device)

        data_module.setup(stage=data_module_setup)
        self.dataloader = data_module.test_dataloader()  # Get the dataloader of the data_module

        self.using_methodology = using_methodology # select what methodology to use
        self.model_idx = model_idx # for create a folder with experiments using this model
        self.execute_only_predictions = execute_only_predictions # for execute only predictions (no plots)
        self.evaluate_this_samples = evaluate_this_samples
    
    def set_points(self):
        """ Teset the accumulated points """
        self.accumulated_coords = np.empty((0, 2), dtype=int) # Nx2 array
        self.accumulated_labels = np.empty((0,), dtype=int) # Array of N length

    def forward(self, batch):
        """ Make inference with model """
        return self.model(batch, multimask_output=self.multimask_output)
    
    def execute_prediction(self, batch, batch_idx):
        if self.using_methodology == 1:
            batch_preds = self.process_v1(batch, batch_idx, "process_v1")
            if not self.execute_only_predictions:
                plot_result_batch_preds(batch_preds=batch_preds, batch_idx=batch_idx, process_func="process_v1", model_idx=self.model_idx, using_methodology=self.using_methodology) # plot results
        elif self.using_methodology == 2:
            batch_preds = self.process_v2(batch, batch_idx, "process_v2")
            if not self.execute_only_predictions:
                plot_result_batch_preds(batch_preds=batch_preds, batch_idx=batch_idx, process_func="process_v2", model_idx=self.model_idx, using_methodology=self.using_methodology) # plot results
        else:
            raise ValueError(f"Informe um valor no parametro using_methodology. Pode ser 1 ou 2.")
        
        # print(batch_preds.shape, type(batch_preds))
        # exit()
        return batch_preds
    
    def predict_step(self, batch, batch_idx):
        """ Test step to use Trainer.test() """

        if self.evaluate_this_samples != None:
            for sample in self.evaluate_this_samples:
                if batch_idx == sample:
                    return self.execute_prediction(batch, batch_idx)
        else:
            return self.execute_prediction(batch, batch_idx)
    
    def predict_dataloader(self):
        return self.dataloader
    
    """ version 1 """
    def process_v1(self, batch, batch_idx, process_func, exec_grad:bool=False, target_layer:str="model.mask_decoder.output_upscaling.3"):
        """ Process the batch sended by test_step """
        
        # processing default dataset of minerva
        if isinstance(batch, list) and len(batch) == 2:
            batch_preds = []
            count_pred = 0
            for sample_idx in range(len(batch[0])):
                image = batch[0][sample_idx]
                label = batch[1][sample_idx]

                # normalize and add 3 channels
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                image = (image * 255).clamp(0, 255).to(torch.uint8)
                image = image.to(self.model.device)

                num_facies = torch.unique(label)  # Identify the classes of the sample
                point_type = 'positive' # The first point (prompt) is positive

                # Initialize a list to store predictions for each number of points
                facie_preds = []

                for facie_idx, facie in enumerate(num_facies):
                    point_preds = [torch.zeros_like(label) for _ in range(self.num_points)]
                    region = (label == facie).to(torch.uint8).to(self.device)
                    real_label = region
                    
                    # Calculate the number of pixels in the region
                    num_pixels = torch.sum(region).item()
                    if num_pixels <= 1:
                        continue # ignore region with 1 pixel or less

                    for point in range(self.num_points):
                        point_coords, point_labels = self.calculate_center_region(region.cpu().numpy(), point_type)

                        # Cria os tensores com os pontos acumulados
                        point_coords_tensor = torch.tensor(point_coords, dtype=torch.long).unsqueeze(0).to(self.device)
                        point_labels_tensor = torch.tensor(point_labels, dtype=torch.long).unsqueeze(0).to(self.device)

                        batch = {
                            'image': image,
                            'label': real_label,
                            'original_size': (int(image.shape[1]), int(image.shape[2])),
                            'point_coords': point_coords_tensor,
                            'point_labels': point_labels_tensor
                        }

                        # calculate GRAD-CAM
                        if exec_grad:
                            # print(f"Executing Grad-CAM in {target_layer}")
                            grad_cam = GradCAM(model=self.model, target_layer=target_layer)
                            cam, output_pred = grad_cam.generate_cam(batch=batch, label=real_label, backward_aproach=2)
                            mask = (output_pred > 0.0).float()

                            # Move all tensors to the same device
                            device = mask.device
                            gt_tensor = torch.tensor(real_label, dtype=torch.float32).to(device)  # Ground truth
                            pred_tensor = mask.squeeze(0).squeeze(0)  # Remover dimensões extras

                            # Ensure the metric is on the same device
                            self.miou_metric = self.miou_metric.to(device)

                            # calculate IoU
                            iou_score = self.miou_metric(pred_tensor, gt_tensor)

                            # the difference need to be between real label and pred, beacause the difference needs a reference (real label)
                            diff, new_point_type = self.calculate_diff_label_pred(label=real_label.cpu().numpy(), pred=mask.squeeze().cpu().numpy())
                            region = torch.tensor(diff).to(self.device)
                            point_type = new_point_type

                            plot_all(
                                image=image.permute(1, 2, 0).cpu().numpy(),
                                label=real_label.cpu().numpy(),
                                pred=mask.squeeze().cpu().numpy(),
                                diff=diff,
                                score=iou_score,
                                point_coords=self.accumulated_coords,
                                point_labels=self.accumulated_labels,
                                i=count_pred,  # Pass the index to the plot_all function
                                batch_idx=batch_idx,
                                process_func=process_func,
                                cam=cam,
                                sugest_filename=f"using_process_v1_pred_with_grad_cam_{target_layer}_backward_aproach_{2}_in_facie_{facie}_using_{point}_points",
                                model_idx=self.model_idx,
                                using_methodology=self.using_methodology
                            )
                            continue # only for calculate grad-cam

                        outputs = self.forward([batch])
                        mask = (outputs > 0.0).float() # remover isso depois

                        # calculate IoU
                        gt_tensor = torch.tensor(real_label, dtype=torch.float32, device=self.model.device)  # Ground truth
                        pred_tensor = mask.squeeze(0).squeeze(0)  # Remover dimensões extras
                        iou_score = self.miou_metric(pred_tensor, gt_tensor)

                        # Accumulate the predictions for the current facie
                        prediction_value = torch.where(mask.bool(), facie, torch.full_like(mask, 7, dtype=facie.dtype)).squeeze()
                        point_preds[point] = (prediction_value, iou_score.item())

                        # the difference need to be between real label and pred, beacause the difference needs a reference (real label)
                        diff, new_point_type = self.calculate_diff_label_pred(label=real_label.cpu().numpy(), pred=mask.squeeze().cpu().numpy())

                        # print(region.cpu().numpy().shape, mask.squeeze().cpu().numpy().shape)
                        if not self.execute_only_predictions:
                            plot_all(
                                image=image.permute(1, 2, 0).cpu().numpy(),
                                label=real_label.cpu().numpy(),
                                pred=mask.squeeze().cpu().numpy(),
                                diff=diff,
                                score=iou_score.item(),
                                point_coords=self.accumulated_coords,
                                point_labels=self.accumulated_labels,
                                i=count_pred,  # Pass the index to the plot_all function
                                batch_idx=batch_idx,
                                process_func=process_func,
                                model_idx=self.model_idx,
                                using_methodology=self.using_methodology
                            )
                        count_pred += 1
                        
                        region = torch.tensor(diff).to(self.device)
                        point_type = new_point_type
                    point_type = 'positive'
                    self.set_points()
                    facie_preds.append(point_preds)
                    # break
                    # exit()
                
                # calculate GRAD-CAM
                if exec_grad:
                    return None
                
                # Transpõe a lista para agrupar as predições de cada ponto
                transposed_preds = list(zip(*facie_preds))
                combined_preds = []
                # Mescla as predições usando prioridade condicional: o valor do pixel só é substituído se o novo valor não for fundo (7)
                for i, group in enumerate(transposed_preds):
                    # group é uma tupla de (mask, iou) de cada facie para o mesmo ponto
                    # Empilha as máscaras: shape (n, H, W)
                    candidate_masks = torch.stack([pred[0] for pred in group], dim=0)
                    # Cria um tensor de IoU para cada predição: shape (n, 1, 1)
                    candidate_ious = torch.tensor([pred[1] for pred in group], device=candidate_masks.device).view(-1, 1, 1)

                    # Se a máscara for fundo (valor 7), defina o IoU como -infinito para que não seja selecionada
                    candidate_ious = torch.where(candidate_masks != 7, candidate_ious, torch.tensor(float('-inf'), device=candidate_masks.device))

                    # Para cada pixel, identifica qual candidato tem o maior IoU
                    max_idx = candidate_ious.argmax(dim=0)  # shape (H, W)
                    # Usa o índice para selecionar o valor correspondente na máscara
                    merged = candidate_masks.gather(0, max_idx.unsqueeze(0)).squeeze(0)

                    # Caso o maior IoU seja -infinito, significa que nenhum candidato era válido (todos fundo)
                    max_ious = candidate_ious.max(dim=0).values
                    merged = torch.where(max_ious == float('-inf'), torch.full_like(merged, 7), merged)
                    combined_preds.append(merged)
                    
                combined_preds = torch.stack(combined_preds, dim=0)
                batch_preds.append(combined_preds)
            batch_preds = torch.stack(batch_preds, dim=0)
            return batch_preds

    """ version 2 """
    def process_v2(self, batch, batch_idx, process_func, exec_grad:bool=False, target_layer:str="model.mask_decoder.output_upscaling.3"):
        """ Process the batch sended by test_step """
        
        # processing default dataset of minerva
        if isinstance(batch, list) and len(batch) == 2:
            batch_preds = []
            count_pred = 0
            for sample_idx in range(len(batch[0])):
                image = batch[0][sample_idx]
                label = batch[1][sample_idx]

                # normalize and add 3 channels
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                image = (image * 255).clamp(0, 255).to(torch.uint8)
                image = image.to(self.model.device)

                num_facies = torch.unique(label)  # Identify the classes of the sample
                point_type = 'positive' # The first point (prompt) is positive

                # Initialize a list to store predictions for each number of points
                facie_preds = []

                for facie_idx, facie in enumerate(num_facies):
                    point_preds = [torch.zeros_like(label) for _ in range(self.num_points)]
                    region = (label == facie).to(torch.uint8).to(self.device)
                    
                    # Calculate the number of pixels in the region
                    num_pixels = torch.sum(region).item()
                    if num_pixels <= 1:
                        continue # ignore region with 1 pixel or less

                    # plot facie with all prompts
                    point_coords_positive, point_coords_negative = self.get_points_in_region(region=region.cpu().numpy(), num_points_positive=self.num_points, num_points_negative=self.num_points)
                    
                    if not exec_grad:
                        # execute only if execute test
                        plot_facie_with_prompts(facie_idx=facie, point_coords_positive=point_coords_positive, point_coords_negative=point_coords_negative, region=region.cpu().numpy(), batch_idx=batch_idx, process_func=process_func, model_idx=self.model_idx, using_methodology=self.using_methodology)
                    
                    pos_idx = 0
                    neg_idx = 0
                    for point in range(self.num_points):
                        # Seleciona o próximo ponto de acordo com o tipo atual (point_type)
                        if point_type == 'positive':
                            novo_ponto = np.array([point_coords_positive[pos_idx]])
                            pos_idx += 1
                            novo_label = np.array([1])   # Define o rótulo para ponto positivo (pode ser ajustado conforme a necessidade)
                        else:  # Quando point_type é negativo
                            novo_ponto = np.array([point_coords_negative[neg_idx]])
                            neg_idx += 1
                            novo_label = np.array([0])   # Define o rótulo para ponto negativo
                        
                        # Acumula os pontos e rótulos
                        self.accumulated_coords = np.vstack([self.accumulated_coords, novo_ponto])
                        self.accumulated_labels = np.hstack([self.accumulated_labels, novo_label])

                        # Cria os tensores com os pontos acumulados
                        point_coords_tensor = torch.tensor(self.accumulated_coords, dtype=torch.long).unsqueeze(0).to(self.device)
                        point_labels_tensor = torch.tensor(self.accumulated_labels, dtype=torch.long).unsqueeze(0).to(self.device)

                        batch = {
                            'image': image,
                            'label': region,
                            'original_size': (int(image.shape[1]), int(image.shape[2])),
                            'point_coords': point_coords_tensor,
                            'point_labels': point_labels_tensor
                        }

                        # calculate GRAD-CAM
                        if exec_grad:
                            # print(f"Executing Grad-CAM in {target_layer}")
                            grad_cam = GradCAM(model=self.model, target_layer=target_layer)
                            cam, output_pred = grad_cam.generate_cam(batch=batch, label=region, backward_aproach=2)
                            mask = (output_pred > 0.0).float()

                            # Move all tensors to the same device
                            device = mask.device
                            gt_tensor = torch.tensor(region, dtype=torch.float32).to(device)  # Ground truth
                            pred_tensor = mask.squeeze(0).squeeze(0)  # Remover dimensões extras

                            # Ensure the metric is on the same device
                            self.miou_metric = self.miou_metric.to(device)

                            # calculate IoU
                            iou_score = self.miou_metric(pred_tensor, gt_tensor)

                            # the difference need to be between real label and pred, beacause the difference needs a reference (real label)
                            diff, new_point_type = self.calculate_diff_label_pred(label=region.cpu().numpy(), pred=mask.squeeze().cpu().numpy())
                            point_type = new_point_type

                            plot_all(
                                image=image.permute(1, 2, 0).cpu().numpy(),
                                label=region.cpu().numpy(),
                                pred=mask.squeeze().cpu().numpy(),
                                diff=diff,
                                score=iou_score,
                                point_coords=self.accumulated_coords,
                                point_labels=self.accumulated_labels,
                                i=count_pred,  # Pass the index to the plot_all function
                                batch_idx=batch_idx,
                                process_func=process_func,
                                cam=cam,
                                sugest_filename=f"using_process_v2_pred_with_grad_cam_{target_layer}_backward_aproach_{2}_in_facie_{facie}_using_{point}_points",
                                model_idx=self.model_idx,
                                using_methodology=self.using_methodology
                            )
                            continue # only for calculate grad-cam

                        # Executing prediction
                        outputs = self.forward([batch])
                        mask = (outputs > 0.0).float() # remover isso depois

                        # calculate IoU
                        gt_tensor = torch.tensor(region, dtype=torch.float32, device=self.model.device)  # Ground truth
                        pred_tensor = mask.squeeze(0).squeeze(0)  # Remover dimensões extras
                        iou_score = self.miou_metric(pred_tensor, gt_tensor)

                        # Accumulate the predictions for the current facie
                        prediction_value = torch.where(mask.bool(), facie, torch.full_like(mask, 7, dtype=facie.dtype)).squeeze()
                        point_preds[point] = (prediction_value, iou_score.item())

                        # the difference need to be between real label and pred, beacause the difference needs a reference (real label)
                        diff, new_point_type = self.calculate_diff_label_pred(label=region.cpu().numpy(), pred=mask.squeeze().cpu().numpy())

                        # print(region.cpu().numpy().shape, mask.squeeze().cpu().numpy().shape)
                        if not self.execute_only_predictions:
                            plot_all(
                                image=image.permute(1, 2, 0).cpu().numpy(),
                                label=region.cpu().numpy(),
                                pred=mask.squeeze().cpu().numpy(),
                                diff=diff,
                                score=iou_score.item(),
                                point_coords=self.accumulated_coords,
                                point_labels=self.accumulated_labels,
                                i=count_pred,  # Pass the index to the plot_all function
                                batch_idx=batch_idx,
                                process_func=process_func,
                                model_idx=self.model_idx,
                                using_methodology=self.using_methodology
                            )
                        count_pred += 1
                        
                        point_type = new_point_type
                    point_type = 'positive'
                    self.set_points()
                    facie_preds.append(point_preds)
                    # break
                
                # calculate GRAD-CAM
                if exec_grad:
                    return None
                        
                # Transpõe a lista para agrupar as predições de cada ponto
                transposed_preds = list(zip(*facie_preds))
                combined_preds = []
                # Mescla as predições usando prioridade condicional: o valor do pixel só é substituído se o novo valor não for fundo (7)
                for i, group in enumerate(transposed_preds):
                    # group é uma tupla de (mask, iou) de cada facie para o mesmo ponto
                    # Empilha as máscaras: shape (n, H, W)
                    candidate_masks = torch.stack([pred[0] for pred in group], dim=0)
                    # Cria um tensor de IoU para cada predição: shape (n, 1, 1)
                    candidate_ious = torch.tensor([pred[1] for pred in group], device=candidate_masks.device).view(-1, 1, 1)

                    # Se a máscara for fundo (valor 7), defina o IoU como -infinito para que não seja selecionada
                    candidate_ious = torch.where(candidate_masks != 7, candidate_ious, torch.tensor(float('-inf'), device=candidate_masks.device))

                    # Para cada pixel, identifica qual candidato tem o maior IoU
                    max_idx = candidate_ious.argmax(dim=0)  # shape (H, W)
                    # Usa o índice para selecionar o valor correspondente na máscara
                    merged = candidate_masks.gather(0, max_idx.unsqueeze(0)).squeeze(0)

                    # Caso o maior IoU seja -infinito, significa que nenhum candidato era válido (todos fundo)
                    max_ious = candidate_ious.max(dim=0).values
                    merged = torch.where(max_ious == float('-inf'), torch.full_like(merged, 7), merged)
                    combined_preds.append(merged)
                    
                combined_preds = torch.stack(combined_preds, dim=0)
                batch_preds.append(combined_preds)
            batch_preds = torch.stack(batch_preds, dim=0)
            return batch_preds

    def get_points_in_region(self, region, num_points_positive=3, num_points_negative=3, offset=9):
        """
        Processa a região binária e retorna os pontos positivos e negativos.

        Returns:
            tuple: Uma tupla contendo:
                - point_coords_positives (np.ndarray): Array de forma (N, 2) com as coordenadas (x, y) dos pontos positivos.
                - negative_points (np.ndarray): Array de forma (M, 2) com as coordenadas (x, y) dos pontos negativos.
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

        # Obter o esqueleto da região para os pontos positivos
        skeleton = skeletonize(region)
        y_indices, x_indices = np.where(skeleton == 1)

        # Se não houver pontos no esqueleto, retorna arrays vazios
        if len(y_indices) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        # Criar lista de pontos (em (x, y))
        skeleton_points = list(zip(x_indices, y_indices))

        # Calcular o centro da região branca usando os pixels brancos (valor 1)
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

        # Processamento dos pontos negativos (a partir dos contornos)
        contours = find_contours(region, 0.5)
        if len(contours) == 0:
            return point_coords_positives, np.empty((0, 2))

        H, W = region.shape
        negative_points = []

        # Iterar sobre cada contorno encontrado
        for cnt in contours:
            cnt = np.array(cnt)  # Cada linha: (y, x)
            if len(cnt) < 2:
                continue

            # Calcular as distâncias acumuladas ao longo do contorno
            diffs = np.diff(cnt, axis=0)
            seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            cumulative_distances = np.concatenate(([0], np.cumsum(seg_lengths)))
            total_length = cumulative_distances[-1]

            # Número de pontos para este contorno
            points_count = min(num_points_negative, len(cnt))
            desired_distances = np.linspace(0, total_length, points_count)

            # Interpolar pontos equidistantes ao longo do contorno
            interpolated_points = []
            for d in desired_distances:
                idx = np.searchsorted(cumulative_distances, d)
                if idx == 0:
                    point = cnt[0]
                elif idx >= len(cnt):
                    point = cnt[-1]
                else:
                    t = (d - cumulative_distances[idx - 1]) / (cumulative_distances[idx] - cumulative_distances[idx - 1])
                    point = (1 - t) * cnt[idx - 1] + t * cnt[idx]
                interpolated_points.append(point)
            interpolated_points = np.array(interpolated_points)  # Cada linha: (y, x)

            # Calcular o centro do contorno (para definir a direção do deslocamento)
            center = np.mean(cnt, axis=0)  # (y, x)
            center_y, center_x = center

            for point in interpolated_points:
                y, x = point  # Coordenadas do ponto do contorno (em y, x)
                # Vetor que vai do centro do contorno até o ponto
                vec = np.array([y - center_y, x - center_x])
                norm_vec = np.linalg.norm(vec)
                if norm_vec == 0:
                    direction = np.array([0, 0])
                else:
                    direction = vec / norm_vec

                # Deslocar o ponto para fora do objeto
                new_point = np.array([y, x]) + offset * direction

                # Ajustar o offset se o ponto extrapolar os limites da imagem
                if direction[0] > 0:
                    allowed_y = (H - 1 - y) / direction[0]
                elif direction[0] < 0:
                    allowed_y = (0 - y) / direction[0]
                else:
                    allowed_y = offset

                if direction[1] > 0:
                    allowed_x = (W - 1 - x) / direction[1]
                elif direction[1] < 0:
                    allowed_x = (0 - x) / direction[1]
                else:
                    allowed_x = offset

                allowed_offset = min(offset, allowed_y, allowed_x)
                new_point = np.array([y, x]) + allowed_offset * direction

                # Garantir que o novo ponto esteja dentro dos limites da imagem
                new_point[0] = np.clip(new_point[0], 0, H - 1)
                new_point[1] = np.clip(new_point[1], 0, W - 1)

                # Se o ponto deslocado não estiver na região preta, procurar o pixel preto mais próximo
                iy, ix = int(round(new_point[0])), int(round(new_point[1]))
                if region[iy, ix] != 0:
                    found = False
                    for r in range(1, 10):
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                ny = np.clip(iy + dy, 0, H - 1)
                                nx = np.clip(ix + dx, 0, W - 1)
                                if region[ny, nx] == 0:
                                    iy, ix = ny, nx
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break

                # Converter para (x, y) para manter a consistência
                negative_points.append((ix, iy))

        # Ordenar pontos negativos pela distância ao centro do componente conectado
        negative_points = sorted(negative_points, key=lambda coord: np.linalg.norm(np.array(coord) - overall_center))

        # Converter para array NumPy
        negative_points = np.array(negative_points)

        return point_coords_positives, negative_points
    
    """ version 1 (calculate the center of region, interativaly with model) """
    def calculate_center_region(self, region: np.array, point_type: str, min_distance: int = 10):
        """
        Calcula o centroide da maior região de pixels brancos de uma imagem binária,
        garantindo que o ponto esteja no centro vertical da região branca.

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
        mask_largest = (labels == largest_label)  # Máscara da maior região branca

        # Encontrar as coordenadas brancas dentro da maior região
        y_indices, x_indices = np.where(mask_largest)

        if len(y_indices) == 0:
            raise ValueError("A maior componente conectada não contém pixels brancos.")

        # Encontrar a coluna central da maior região branca
        center_x = np.median(x_indices).astype(int)  # Ponto central em X

        # Encontrar o centro vertical real
        y_in_column = y_indices[x_indices == center_x]
        if len(y_in_column) > 0:
            center_y = y_in_column[len(y_in_column) // 2]  # Pega um valor real no centro da faixa vertical

        new_coords = np.array([[center_x, center_y]])

        # Verificar se o ponto está muito próximo dos anteriores
        if self.accumulated_coords.shape[0] > 0:
            distances = np.sqrt(np.sum((self.accumulated_coords - new_coords) ** 2, axis=1))
            if np.any(distances < min_distance):
                # print("Ponto muito próximo ao anterior, deslocando horizontalmente...")
                
                # Tentar deslocar o ponto horizontalmente dentro da região branca
                region_height, region_width = region.shape
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

        diff_binary = np.zeros(label.shape, dtype=np.uint8) # [H,W]

        # Comparar as áreas
        if area_outward > area_inward:
            diff_binary[mask_outward] = 1
            point_type = 'positive'
        else:
            diff_binary[mask_inward] = 1
            point_type = 'negative'
        
        return diff_binary, point_type
    
    @staticmethod
    def run(
        model:L.LightningModule, 
        data_module:L.LightningDataModule,
        accelerator:str = "gpu",
        devices:int = 1,
        **kwargs
    ):
        """ Static method to run AUC Inferencer """
        calculator = AUCInferencer(model, data_module, **kwargs)

        # calculates the grad-cam
        # if not kwargs['execute_only_predictions']:
        #     print(f"***** Using methodology process_v{kwargs['using_methodology']} *****")
        #     for batch_idx, batch in enumerate(tqdm(calculator.dataloader, desc="Processing Grad-CAM in batches")):
        #         # apply grad-cam only specific images
        #         # for sample in kwargs['evaluate_this_samples']:
        #         #     if batch_idx == sample:
        #         if batch_idx == 0 or batch_idx == 199:
        #             if kwargs['using_methodology'] == 1:
        #                 calculator.process_v1(batch=batch, batch_idx=batch_idx, process_func="process_v1", exec_grad=True, target_layer="model.image_encoder.neck.2")
        #             elif kwargs['using_methodology'] == 2:
        #                 calculator.process_v2(batch=batch, batch_idx=batch_idx, process_func="process_v2", exec_grad=True, target_layer="model.image_encoder.neck.2")
        #             else:
        #                 raise ValueError(f"Informe um valor no parametro using_methodology. Pode ser 1 ou 2.")

        # calculates the inference
        trainer = L.Trainer(accelerator=accelerator, devices=[devices])
        return trainer.predict(calculator)