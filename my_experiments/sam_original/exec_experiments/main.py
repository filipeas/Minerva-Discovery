import os
import shutil
import argparse
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from minerva.models.finetune_adapters import LoRA
from minerva.models.nets.image.sam import Sam
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from data_module import DataModule, Transpose, Padding

def _init_experiment(
        name,
        num_classes,
        batch_size,
        alpha,
        rank,
        N,
        epochs,
        height_image, 
        width_image,
        data_ratios,
        aways_freeze_this_component,
        apply_in_components,
        multimask_output,
        vit_model,
        checkpoint_path,
        train_path,
        annotation_path,
        config_path,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(os.getcwd()) / f"results/{name}_experiment_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "experiment_log.csv"  # Log em CSV
    log_data = []  # Lista para armazenar os dados do experimento
    if config_path:
        shutil.copy(config_path, save_dir / "config_copy.json")
    
    for ratio in data_ratios:
        print(f"Executando experimentos com {int(ratio * 100)}% dos dados...")

        # Loop externo: Aplicar fine-tuning, freeze ou adapter no image_encoder
        for image_encoder_method in apply_in_components:
            print(f"\n>> Método no Image Encoder: {image_encoder_method}")
            
            # Loop interno: Variar o mask_decoder com fine-tuning, freeze e adapter
            for mask_decoder_method in apply_in_components:
                if image_encoder_method == 'freeze' and mask_decoder_method == 'freeze':
                    print("freeze no image encoder e mask decoder deve ser ignorado")
                    continue

                print(f"   >> Método no Mask Decoder: {mask_decoder_method}")

                for _ in tqdm(range(N), desc=f"Image: {image_encoder_method}, Mask: {mask_decoder_method}"):
                    # Inicializar dicionários de configuração
                    apply_freeze = {
                        aways_freeze_this_component: True, # Sempre congelar prompt_encoder
                        "image_encoder": True, # freeze por padrão
                        "mask_decoder": True # freeze por padrão
                    }
                    apply_adapter = {} # Adapters vazios inicialmente
                    
                    # Configurar o image_encoder conforme o método externo
                    if image_encoder_method == "fine_tuning":
                        apply_freeze["image_encoder"] = False  # Descongela para fine-tuning
                    elif image_encoder_method == "adapter":
                        apply_adapter["image_encoder"] = LoRA
                    
                    # Configurar o mask_decoder conforme o método interno
                    if mask_decoder_method == "fine_tuning":
                        apply_freeze["mask_decoder"] = False  # Descongela para fine-tuning
                    elif mask_decoder_method == "adapter":
                        apply_adapter["mask_decoder"] = LoRA
                    
                    """ starting training """
                    val_loss_epoch, train_loss_epoch, test_loss_epoch, val_mIoU, train_mIoU, test_mIoU = execute_train(height_image, width_image, ratio, alpha, rank, train_path, annotation_path, multimask_output, batch_size, vit_model, checkpoint_path, num_classes, apply_freeze, apply_adapter, epochs)

                    log_data.append({
                        "ratio": ratio,
                        "image_encoder_method": image_encoder_method,
                        "mask_decoder_method": mask_decoder_method,
                        "val_loss_epoch": val_loss_epoch,
                        "train_loss_epoch": train_loss_epoch,
                        "test_loss_epoch": test_loss_epoch,
                        "val_mIoU": val_mIoU,
                        "train_mIoU": train_mIoU,
                        "test_mIoU": test_mIoU
                    })
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
        multimask_output,
        batch_size,
        vit_model,
        checkpoint_path,
        num_classes,
        apply_freeze,
        apply_adapter,
        epochs
):
    data_module = DataModule(
        train_path=train_path,
        annotations_path=annotation_path,
        transforms=Padding(height_image, width_image),
        # transform_coords_input={'point_coords': None, 'point_labels': None},
        multimask_output=multimask_output,
        batch_size=batch_size,
        data_ratio=ratio
    )

    # debug
    # def get_train_dataloader(data_module):
    #     data_module.setup("fit")
    #     return data_module.train_dataloader()

    # print("Total batches: ", len(get_train_dataloader(data_module)))

    # train_batch = next(iter(get_train_dataloader(data_module)))
    # print(f"Train batch image (X) shape: {train_batch[0]['image'].shape}")
    # print(f"Train batch label (Y) shape: {train_batch[0]['label'].shape}")
    # print(f"Train batch label (original_size) shape: {train_batch[0]['original_size']}")
    # print(f"multimask_output: {train_batch[0]['multimask_output']}")

    # for idx, batch in enumerate(get_train_dataloader(data_module)):
    #     print(f"Batch {idx}:")
    #     print(f"Tipo do batch: {type(batch)}")
    #     print(f"Tamanho do batch: {len(batch)}")  # Deve ser igual ao batch_size
    #     print("Estrutura do primeiro item do batch:")
    #     # print(batch[0])  # Exibe o primeiro dicionário do batch
    #     print(f"Shape da imagem no primeiro item: {batch[0]['image'].shape}")
    #     print(20*'-')
    #     print(f"Train batch image (X) shape: {batch[0]['image'].shape}")
    #     print(f"Train batch label (Y) shape: {batch[0]['label'].shape}")
    #     print(f"Train batch label (original_size) shape: {batch[0]['original_size']}")
    #     print(f"multimask_output: {batch[0]['multimask_output']}")
    #     break  # Para após o primeiro batch
    # print(f"O Batch (de tamanho {len(train_batch)}) possui: {train_batch[0]['image'].shape[0]} canais, {train_batch[0]['image'].shape[1]} altura e {train_batch[0]['image'].shape[2]} largura.")

    model = Sam(
        vit_type=vit_model,
        checkpoint=checkpoint_path,
        num_multimask_outputs=num_classes, # default: 3
        iou_head_depth=num_classes, # default: 3
        apply_freeze=apply_freeze,
        apply_adapter=apply_adapter,
        lora_rank=rank,
        lora_alpha=alpha,
    )

    # current_date = datetime.now().strftime("%Y-%m-%d")
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",
    #     dirpath="./checkpoints",
    #     filename=f"sam-{current_date}-{{epoch:02d}}-{{val_loss:.2f}}",
    #     save_top_k=1,
    #     mode="min",
    # )
    
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False
        # callbacks=[checkpoint_callback],
    )

    # pipeline
    pipeline = SimpleLightningPipeline(model=model, trainer=trainer, save_run_status=True)
    
    pipeline.run(data=data_module, task="fit")
    val_loss = trainer.callback_metrics['val_loss']
    val_mIoU = trainer.callback_metrics['val_mIoU']
    train_loss = trainer.callback_metrics['train_loss']
    train_mIoU = trainer.callback_metrics['train_mIoU']
    
    test = pipeline.run(data=data_module, task="test")
    test_loss = test[0]['test_loss']
    test_mIoU = test[0]['test_mIoU']

    return val_loss, train_loss, test_loss, val_mIoU, train_mIoU, test_mIoU

""" main function """
if __name__ == "__main__":
    print("___START EXPERIMENT___")

    # Inicializa o analisador de argumentos
    parser = argparse.ArgumentParser(description="Script for experiments with SAM")
    parser.add_argument('--config', type=str, help="Caminho para o arquivo de configuração JSON", required=True)
    
    args = parser.parse_args()

    # Carregar configurações do JSON
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extrair os parâmetros do JSON
    name = config["name"]
    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    alpha = config["alpha"]
    rank = config["rank"]
    n = config["n"]
    epochs = config["epochs"]
    height_image = config['height_image']
    width_image = config['width_image']
    data_ratios = config["data_ratios"]
    aways_freeze_this_component = config['aways_freeze_this_component']
    apply_in_components = config['apply_in_components']
    multimask_output = config['multimask_output']
    vit_model = config["vit_model"]
    checkpoint_sam = config["checkpoint_sam"]
    train_path = config["train_path"]
    annotation_path = config["annotation_path"]

    if not isinstance(data_ratios, list):
        raise ValueError("O parâmetro 'data_ratios' no JSON precisa ser uma lista.")
    if not isinstance(apply_in_components, list):
        raise ValueError("O parâmetro 'apply_in_components' no JSON precisa ser uma lista.")

    # Exibir parâmetros em formato de tabela
    print(20*'-')
    print("\n--- Parâmetros de Configuração ---")
    print(f"{'Parâmetro':<20} {'Valor'}")
    print("-" * 40)
    print(f"{'name':<20} {name}")
    print(f"{'num_classes':<20} {num_classes}")
    print(f"{'batch_size':<20} {batch_size}")
    print(f"{'alpha':<20} {alpha}")
    print(f"{'rank':<20} {rank}")
    print(f"{'n':<20} {n}")
    print(f"{'epochs':<20} {epochs}")
    print(f"{'height_image':<20} {height_image}")
    print(f"{'width_image':<20} {width_image}")
    print(f"{'data_ratios':<20} {data_ratios}")
    print(f"{'aways_freeze_this_component':<20} {aways_freeze_this_component}")
    print(f"{'apply_in_components':<20} {apply_in_components}")
    print(f"{'multimask_output'}:<20 {multimask_output}")
    print(f"{'vit_model':<20} {vit_model}")
    print(f"{'checkpoint_sam':<20} {checkpoint_sam}")
    print(f"{'train_path':<20} {train_path}")
    print(f"{'annotation_path':<20} {annotation_path}")
    print(20*'-')

    _init_experiment(
        name=name,
        num_classes=num_classes,
        batch_size=batch_size,
        alpha=alpha,
        rank=rank,
        N=n, 
        epochs=epochs, 
        height_image=height_image, 
        width_image=width_image,
        data_ratios=data_ratios,
        aways_freeze_this_component=aways_freeze_this_component,
        apply_in_components=apply_in_components,
        multimask_output=multimask_output,
        vit_model=vit_model,
        checkpoint_path=checkpoint_sam,
        train_path=train_path,
        annotation_path=annotation_path,
        config_path=args.config
    )

    print("___END OF EXPERIMENT___")
    print("Good Night ;p")