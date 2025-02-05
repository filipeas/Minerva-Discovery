import os
import shutil
import argparse
import json
from pathlib import Path
import pandas as pd
import lightning as L
from torchmetrics import JaccardIndex
from minerva.models.nets.image.sam import Sam
from datetime import datetime
from data_module import DataModule, Padding
from tqdm import tqdm
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from dataset_for_test import AUC_calculate_v1, DataModule_for_AUC
import torch

def _init_experiment(
        name,
        num_classes,
        num_points,
        batch_size,
        alpha,
        rank,
        N,
        epochs,
        height_image, 
        width_image,
        data_ratios,
        vit_model,
        checkpoint_path,
        train_path,
        annotation_path,
        config_path,
        apply_special_test,
        train_path_special_test,
        annotation_path_special_test,
        gpu_index
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(os.getcwd()) / f"results_with_prompt/{name}_experiment_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "experiment_log.csv"  # Log em CSV
    log_data = []  # Lista para armazenar os dados do experimento
    if config_path:
        shutil.copy(config_path, save_dir / "config_copy.json")
    
    for ratio in data_ratios:
        print(f"Executing experiments with {int(ratio * 100)}% of data...")
        for experiment_num in tqdm(range(N)):
            """ starting training """
            val_loss_epoch, train_loss_epoch, test_loss_epoch, val_mIoU, train_mIoU, test_mIoU = execute_train(
                height_image=height_image, 
                width_image=width_image, 
                ratio=ratio, 
                alpha=alpha, 
                rank=rank, 
                train_path=train_path, 
                annotation_path=annotation_path, 
                batch_size=batch_size, 
                vit_model=vit_model, 
                checkpoint_path=checkpoint_path, 
                num_classes=num_classes, 
                epochs=epochs, 
                num_points=num_points,
                experiment_num=experiment_num,
                save_dir=save_dir,
                apply_special_test=apply_special_test,
                train_path_special_test=train_path_special_test,
                annotation_path_special_test=annotation_path_special_test,
                gpu_index=gpu_index)

            log_data.append({
                "ratio": ratio,
                "experiment_num": experiment_num,
                "num_points": num_points,
                "val_loss_epoch": val_loss_epoch,
                "train_loss_epoch": train_loss_epoch,
                "test_loss_epoch": test_loss_epoch,
                "val_mIoU": val_mIoU,
                "train_mIoU": train_mIoU,
                "test_mIoU": test_mIoU
            })
    # saving progress
    df = pd.DataFrame(log_data)
    df.to_csv(log_file, index=False)
    print(f"Logs salvos em: {log_file}")

def execute_train(
        height_image, 
        width_image,
        ratio,
        alpha,
        rank,
        train_path,
        annotation_path,
        batch_size,
        vit_model,
        checkpoint_path,
        num_classes,
        epochs,
        num_points,
        experiment_num,
        save_dir,
        apply_special_test,
        train_path_special_test,
        annotation_path_special_test,
        gpu_index
):
    data_module = DataModule(
        train_path=train_path,
        annotations_path=annotation_path,
        transforms=Padding(height_image, width_image),
        batch_size=batch_size,
        data_ratio=ratio,
        num_points=num_points
    )

    model = Sam(
        vit_type=vit_model,
        checkpoint=checkpoint_path,
        num_multimask_outputs=num_classes, # default: 3
        iou_head_depth=num_classes, # default: 3
        apply_freeze={"image_encoder": False, "prompt_encoder": False, "mask_decoder": False},
        # apply_adapter=apply_adapter,
        lora_alpha=alpha,
        lora_rank=rank,
        train_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
        val_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
        test_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)}
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[gpu_index],  # Use o índice da GPU especificado,
        logger=False,
        enable_checkpointing=False
    )

    # pipeline
    pipeline = SimpleLightningPipeline(model=model, trainer=trainer, save_run_status=True)
    
    pipeline.run(data=data_module, task="fit")
    val_loss = trainer.callback_metrics['val_loss']
    val_mIoU = trainer.callback_metrics['val_mIoU']
    train_loss = trainer.callback_metrics['train_loss']
    train_mIoU = trainer.callback_metrics['train_mIoU']
    
    test = pipeline.run(data=data_module, task="test")
    test_loss = test[0]['test_loss_epoch']
    test_mIoU = test[0]['test_mIoU_epoch']

    # executing test for colect AUC
    data_module_for_auc = DataModule_for_AUC(
        train_path=train_path if apply_special_test == False else train_path_special_test,
        annotations_path=annotation_path if apply_special_test == False else annotation_path_special_test,
        transforms=Padding(height_image, width_image),
        batch_size=batch_size
    )

    data_module_for_auc.setup(stage='test')  # Configura os dados para inferência
    test_dataloader = data_module_for_auc.test_dataloader()

    if torch.cuda.is_available():
        device = f"cuda:{gpu_index}"  # Use o índice da GPU especificado
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device: ", device)

    model.to(device)
    model.eval()
    
    algorithm = AUC_calculate_v1(
        model=model,
        dataloader=test_dataloader,
        num_points=10,
        multimask_output=False,
        experiment_num=experiment_num,
        save_dir=save_dir
    )

    algorithm.process()

    return val_loss, train_loss, test_loss, val_mIoU, train_mIoU, test_mIoU

""" main function """
if __name__ == "__main__":
    print("___START EXPERIMENT___")

    # Inicializa o analisador de argumentos
    parser = argparse.ArgumentParser(description="Script for experiments with SAM")
    parser.add_argument('--config', type=str, help="Path to file with configurations - file need to be JSON type", required=True)
    args = parser.parse_args()

    # Carregar configurações do JSON
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extrair os parâmetros do JSON
    name = config["name"]
    num_classes = config["num_classes"]
    num_points = config["num_points"]
    batch_size = config["batch_size"]
    alpha = config["alpha"]
    rank = config["rank"]
    n = config["n"]
    epochs = config["epochs"]
    height_image = config['height_image']
    width_image = config['width_image']
    data_ratios = config["data_ratios"]
    vit_model = config["vit_model"]
    checkpoint_sam = config["checkpoint_sam"]
    train_path = config["train_path"]
    annotation_path = config["annotation_path"]
    apply_special_test = config["apply_special_test"]
    train_path_special_test = config["train_path_special_test"]
    annotation_path_special_test = config["annotation_path_special_test"]
    gpu_index = config["gpu_index"]

    if not isinstance(data_ratios, list):
        raise ValueError("O parâmetro 'data_ratios' no JSON precisa ser uma lista.")

    # Exibir parâmetros em formato de tabela
    print(20*'-')
    print("\n--- Parâmetros de Configuração ---")
    print(f"{'Parâmetro':<20} {'Valor'}")
    print("-" * 40)
    print(f"{'name':<20} {name}")
    print(f"{'num_classes':<20} {num_classes}")
    print(f"{'num_points':<20} {num_points}")
    print(f"{'batch_size':<20} {batch_size}")
    print(f"{'alpha':<20} {alpha}")
    print(f"{'rank':<20} {rank}")
    print(f"{'n':<20} {n}")
    print(f"{'epochs':<20} {epochs}")
    print(f"{'height_image':<20} {height_image}")
    print(f"{'width_image':<20} {width_image}")
    print(f"{'data_ratios':<20} {data_ratios}")
    print(f"{'vit_model':<20} {vit_model}")
    print(f"{'checkpoint_sam':<20} {checkpoint_sam}")
    print(f"{'train_path':<20} {train_path}")
    print(f"{'annotation_path':<20} {annotation_path}")
    print(f"{'apply_special_test':<20} {apply_special_test}")
    print(f"{'train_path_special_test':<20} {train_path_special_test}")
    print(f"{'annotation_path_special_test':<20} {annotation_path_special_test}")
    print(f"{'gpu_index':<20} {gpu_index}")
    print(20*'-')

    _init_experiment(
        name=name,
        num_classes=num_classes,
        num_points=num_points,
        batch_size=batch_size,
        alpha=alpha,
        rank=rank,
        N=n, 
        epochs=epochs, 
        height_image=height_image, 
        width_image=width_image,
        data_ratios=data_ratios,
        vit_model=vit_model,
        checkpoint_path=checkpoint_sam,
        train_path=train_path,
        annotation_path=annotation_path,
        config_path=args.config,
        apply_special_test=apply_special_test,
        train_path_special_test=train_path_special_test,
        annotation_path_special_test=annotation_path_special_test,
        gpu_index=gpu_index
    )

    print("___END OF EXPERIMENT___")
    print("Good Night ;p")