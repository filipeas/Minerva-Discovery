import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import get_trainer_pipeline
from init_experiment import init_sam
from common import get_data_module_for_sam
from dataset import DatasetForSAM_experiment_1, DatasetForSAM_experiment_2, DatasetForSAM_experiment_3

def train(
        data_module,
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

    checkpoint_path = os.path.join(log_dir, model_name, dataset_name, "checkpoints", "last.ckpt")
    if os.path.exists(checkpoint_path):
        print("Last chackpoint already exists.")
        return

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

    pipeline.run(data_module, task="fit")

    print(f"Checkpoint saved at {pipeline.trainer.checkpoint_callback.last_model_path}")

def check_dataset(
        dataset_class,
        dataset_params,
        root_data_dir:str, 
        root_annotation_dir:str, 
        model_info:type, 
        filename:str, 
        img_size:str=(1006, 590),
        single_channel:bool=False, 
        batch_size:int=3, 
        seed:int=42):
    print("Executing test for get_data_module function")

    data_module = get_data_module_for_sam(
        dataset_class=dataset_class,
        dataset_params=dataset_params,
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        img_size=img_size,
        single_channel=single_channel,
        batch_size=batch_size,
        seed=seed,
        num_workers=os.cpu_count()
    )

    data_module.setup("test")
    check_dataloader = data_module.test_dataloader()
    print("Total batches: ", len(check_dataloader))

    # Verificar todas as amostras no dataloader
    incorrect_samples = []
    expected_shape = (1006, 590)

    for i, batch in enumerate(check_dataloader):
        for j, sample in enumerate(batch):
            image = sample['image']
            label = sample['label']

            # Obter as dimensões da imagem e do label (desconsiderando os canais)
            img_shape = image.shape[-2:]  # Últimas duas dimensões (altura, largura)
            label_shape = label.shape[-2:]  # Últimas duas dimensões (altura, largura)
            # print(img_shape, label_shape)

            if img_shape != expected_shape or label_shape != expected_shape:
                incorrect_samples.append((i, j, img_shape, label_shape))
                raise ValueError(f"Sample {j} in batch {i} has incorrect shape: Image {img_shape}, Label {label_shape}")

    sample_index = 50  # a sample for test
    check_batch = [check_dataloader.dataset[sample_index]]

    print(f"Train batch image (X) shape: {check_batch[0]['image'].shape} - type: {type(check_batch[0]['image'])}")
    print(f"Train batch label (Y) shape: {check_batch[0]['label'].shape} - type: {type(check_batch[0]['label'])}")
    print(f"Train batch label (original_size) shape: {check_batch[0]['original_size']} - type: {type(check_batch[0]['original_size'])}")

    if model_info['model_name'] == 'sam_vit_b_experiment_3' or model_info['model_name'] == 'sam_vit_b_experiment_4':
        print(f"Train batch point_coords shape: {check_batch[0]['point_coords'].shape} - type: {type(check_batch[0]['point_coords'])}")
        print(f"Train batch point_labels shape: {check_batch[0]['point_labels'].shape} - type: {type(check_batch[0]['point_labels'])}")
    
    print(f"O Batch (de tamanho {len(check_batch)}) possui: {check_batch[0]['image'].shape[0]} canais, {check_batch[0]['image'].shape[1]} altura e {check_batch[0]['image'].shape[2]} largura.")

    # Obtendo a imagem e a label do batch
    if model_info['model_name'] == 'sam_vit_b_experiment_3' or model_info['model_name'] == 'sam_vit_b_experiment_4':
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

    if model_info['model_name'] == 'sam_vit_b_experiment_3' or model_info['model_name'] == 'sam_vit_b_experiment_4':
        # Plotando os pontos na imagem
        for point in points:
            for y, x in point:
                axes[0].scatter(y, x, color='red', s=50, marker='x', label='Ponto')

    # Label (provavelmente uma máscara ou rótulo binário)
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title("Label")
    axes[1].axis('off')

    if model_info['model_name'] == 'sam_vit_b_experiment_3' or model_info['model_name'] == 'sam_vit_b_experiment_4':
        # plotando os pontos na label
        for point in points:
            for y, x in point:
                axes[1].scatter(y, x, color='red', s=50, marker='x', label='Ponto')

    os.makedirs('tmp', exist_ok=True)
    output_path = f"tmp/{filename}.png"
    plt.savefig(output_path)
    print(f"The image {filename} is saved in: {output_path}")
    plt.close(fig)

    return data_module

if __name__ == "__main__":
    root_data_dir = "/workspaces/Minerva-Discovery/shared_data/seam_ai_datasets/seam_ai/images"
    root_annotation_dir = "/workspaces/Minerva-Discovery/shared_data/seam_ai_datasets/seam_ai/annotations"
    ckpt_file = "/workspaces/Minerva-Discovery/shared_data/weights_sam/checkpoints_sam/sam_vit_b_01ec64.pth"
    ckpt_file_exp2 = "/workspaces/Minerva-Discovery/my_experiments/sam_original/evaluate_experiments/parihaka/tmp/logs/sam_vit_b_experiment_2/seam_ai/checkpoints/last.ckpt"
    img_size = (1006,590)

    print("Loading model for experiment 1")
    model_info_experiment_1 = init_sam(
        model_name="sam_vit_b_experiment_1",
        ckpt_file=ckpt_file,
        another_args={
            "apply_freeze": {"prompt_encoder": True, "image_encoder": False, "mask_decoder": False},
            "apply_adapter": {}
        },
        num_classes = 6
    )
    print("Loading model for experiment 2")
    model_info_experiment_2 = init_sam(
        model_name="sam_vit_b_experiment_2",
        ckpt_file=ckpt_file,
        num_classes = 3
    )
    print("Loading model for experiment 3")
    model_info_experiment_3 = init_sam(
        model_name="sam_vit_b_experiment_3",
        ckpt_file=ckpt_file,
        another_args={
            "apply_freeze": {"prompt_encoder": False, "image_encoder": False, "mask_decoder": False},
            "apply_adapter": {}
        },
        num_classes = 3
    )
    # special experiment (using checkpoint 2 in experiment 3)
    print("Loading model for experiment 4")
    model_info_experiment_4 = init_sam(
        model_name="sam_vit_b_experiment_4",
        ckpt_file=ckpt_file_exp2,
        another_args={
            "apply_freeze": {"prompt_encoder": False, "image_encoder": False, "mask_decoder": False},
            "apply_adapter": {}
        },
        num_classes = 3,
        apply_load_from_checkpoint = True
    )

    print("-"*80)
    print("Checking data module for experiment 1")
    data_module_exp1 = check_dataset(dataset_class=DatasetForSAM_experiment_1, dataset_params={}, root_data_dir=root_data_dir, root_annotation_dir=root_annotation_dir, model_info=model_info_experiment_1, filename="experiment_1", img_size=img_size)
    print("-"*20)
    print("Checking data module for experiment 2")
    data_module_exp2 = check_dataset(dataset_class=DatasetForSAM_experiment_2, dataset_params={"select_facie": 1}, root_data_dir=root_data_dir, root_annotation_dir=root_annotation_dir, model_info=model_info_experiment_2, filename="experiment_2", img_size=img_size)
    print("-"*20)
    print("Checking data module for experiment 3")
    data_module_exp3 = check_dataset(dataset_class=DatasetForSAM_experiment_3, dataset_params={"num_points": 3}, root_data_dir=root_data_dir, root_annotation_dir=root_annotation_dir, model_info=model_info_experiment_3, filename="experiment_3", img_size=img_size)
    print("-"*20)
    print("Checking data module for experiment 4")
    data_module_exp4 = check_dataset(dataset_class=DatasetForSAM_experiment_3, dataset_params={"num_points": 3}, root_data_dir=root_data_dir, root_annotation_dir=root_annotation_dir, model_info=model_info_experiment_4, filename="experiment_4", img_size=img_size)
    print("-"*80)

    print("\/"*40)

    print("-"*80)
    print("Training experiment 1")
    train(data_module=data_module_exp1, model_info=model_info_experiment_1, devices=1, is_debug=False)
    del data_module_exp1
    print("-"*20)
    print("Training experiment 2")
    train(data_module=data_module_exp2, model_info=model_info_experiment_2, devices=1, is_debug=False)
    del data_module_exp2
    print("-"*20)
    print("Training experiment 3")
    train(data_module=data_module_exp3, model_info=model_info_experiment_3, devices=1, is_debug=False)
    del data_module_exp3
    print("-"*20)
    print("Training experiment 4")
    train(data_module=data_module_exp4, model_info=model_info_experiment_4, devices=1, is_debug=False)
    print("-"*80)

    print("Everything is ok. Bye!")