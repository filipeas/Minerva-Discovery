import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import get_trainer_pipeline
from init_experiment import init_experiment_1, init_experiment_2, init_experiment_3

def train(
    model_info:type, # init_experiment (can be experiment 1, 2 or 3)
    is_debug:bool = True, # If True, only 3 batch will be processed for 3 epochs
    seed:int = 42, # Seed for reproducibility
    num_epochs:int = 100,  # Number of epochs to train
    accelerator:str = "gpu",  # CPU or GPU
    devices:int = 1,  # Index of GPU
):
    model_name = model_info['model_name']  # Model name (just identifier)
    dataset_name = "seam_ai"  # Dataset name (just identifier)

    log_dir = "tmp/logs"  # Directory to save logs
    os.makedirs(log_dir, exist_ok=True)

    pipeline = get_trainer_pipeline(
        model=model_info['model'],
        model_name=model_name,
        dataset_name=dataset_name,
        log_dir=log_dir,
        num_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        is_debug=is_debug,
        seed=seed,
    )

    pipeline.run(model_info['data_module'], task="fit")

    print(f"Checkpoint saved at {pipeline.trainer.checkpoint_callback.last_model_path}")

def check_dataset(model_info:type, filename:str):
    print("Executing test for get_data_module function")

    model_info['data_module'].setup("test")
    check_dataloader = model_info['data_module'].test_dataloader()
    print("Total batches: ", len(check_dataloader))

    sample_index = 50  # a sample for test
    check_batch = [check_dataloader.dataset[sample_index]]

    print(f"Train batch image (X) shape: {check_batch[0]['image'].shape} - type: {type(check_batch[0]['image'])}")
    print(f"Train batch label (Y) shape: {check_batch[0]['label'].shape} - type: {type(check_batch[0]['label'])}")
    print(f"Train batch label (original_size) shape: {check_batch[0]['original_size']} - type: {type(check_batch[0]['original_size'])}")

    if model_info['model_name'] == 'sam_vit_b_experiment_3':
        print(f"Train batch point_coords shape: {check_batch[0]['point_coords'].shape} - type: {type(check_batch[0]['point_coords'])}")
        print(f"Train batch point_labels shape: {check_batch[0]['point_labels'].shape} - type: {type(check_batch[0]['point_labels'])}")
    
    print(f"O Batch (de tamanho {len(check_batch)}) possui: {check_batch[0]['image'].shape[0]} canais, {check_batch[0]['image'].shape[1]} altura e {check_batch[0]['image'].shape[2]} largura.")

    # Obtendo a imagem e a label do batch
    if model_info['model_name'] == 'sam_vit_b_experiment_3':
        points = check_batch[0]['point_coords']
    image = check_batch[0]['image']#.squeeze(0)  # Remover a dimensão do batch (1, 3, 1006, 590) -> (3, 1006, 590)
    label = check_batch[0]['label']#.squeeze(0)  # Remover a dimensão do batch (1, 1, 1006, 590) -> (1, 1006, 590)

    # Transformando para formato adequado para matplotlib
    image = np.transpose(image, (1, 2, 0))  # (3, 1006, 590) -> (1006, 590, 3)
    label = np.squeeze(label, axis=0)  # (1, 1006, 590) -> (1006, 590)

    # Plotando a imagem e a label
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Imagem original
    axes[0].imshow(image)
    axes[0].set_title("Imagem")
    axes[0].axis('off')

    if model_info['model_name'] == 'sam_vit_b_experiment_3':
        # Plotando os pontos na imagem
        for point in points:
            for y, x in point:
                axes[0].scatter(y, x, color='red', s=50, marker='x', label='Ponto')

    # Label (provavelmente uma máscara ou rótulo binário)
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title("Label")
    axes[1].axis('off')

    if model_info['model_name'] == 'sam_vit_b_experiment_3':
        # plotando os pontos na label
        for point in points:
            for y, x in point:
                axes[1].scatter(y, x, color='red', s=50, marker='x', label='Ponto')

    os.makedirs('tmp', exist_ok=True)
    output_path = f"tmp/{filename}.png"
    plt.savefig(output_path)
    print(f"The image {filename} is saved in: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    root_data_dir = "/workspaces/Minerva-Discovery/shared_data/seam_ai_datasets/seam_ai/images"
    root_annotation_dir = "/workspaces/Minerva-Discovery/shared_data/seam_ai_datasets/seam_ai/annotations"
    ckpt_file = "/workspaces/Minerva-Discovery/shared_data/weights_sam/checkpoints_sam/sam_vit_b_01ec64.pth"

    model_info_experiment_1 = init_experiment_1(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        ckpt_file=ckpt_file,
        img_size=(1006, 590),
        batch_size=3,
        seed=42,
        single_channel=False,
    )
    model_info_experiment_2 = init_experiment_2(
        facie=1,
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        ckpt_file=ckpt_file,
        img_size=(1006, 590),
        batch_size=3,
        seed=42,
        single_channel=False,
    )
    model_info_experiment_3 = init_experiment_3(
        num_points=3,
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        ckpt_file=ckpt_file,
        img_size=(1006, 590),
        batch_size=2,
        seed=42,
        single_channel=False,
    )

    print("-"*80)
    print("Checking data module for experiment 1")
    check_dataset(model_info=model_info_experiment_1, filename="experiment_1")
    print("-"*20)
    print("Checking data module for experiment 2")
    check_dataset(model_info=model_info_experiment_2, filename="experiment_2")
    print("-"*20)
    print("Checking data module for experiment 3")
    check_dataset(model_info=model_info_experiment_3, filename="experiment_3")
    print("-"*80)

    print("\/"*40)

    print("-"*80)
    print("Training experiment 1")
    train(model_info=model_info_experiment_1, devices=1, is_debug=False)
    print("-"*20)
    print("Training experiment 2")
    train(model_info=model_info_experiment_2, devices=1, is_debug=False)
    print("-"*20)
    print("Training experiment 3")
    train(model_info=model_info_experiment_3, devices=1, is_debug=False)
    print("-"*80)

    print("Everything is ok. Bye!")