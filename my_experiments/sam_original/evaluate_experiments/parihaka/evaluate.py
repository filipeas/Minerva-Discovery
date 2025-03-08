import os
import numpy as np
import traceback
import torch
import lightning as L
from minerva.models.loaders import FromPretrained
from typing import Tuple
from pathlib import Path
from common import get_data_module
from init_experiment import init_sam
from auc_inferencer import AUCInferencer

def load_model(model, ckpt):
    return FromPretrained(model, ckpt, strict=False)

def load_model_from_info(model_info):
    model = model_info["model"]
    ckpt_file = model_info["ckpt_file"]
    return load_model(model, ckpt_file)

def load_model_and_data_module(
    root_data_dir:str,
    root_annotation_dir:str,
    model_instantiator_func,
    batch_size:int = 1,
    n_classes:int = 6,
    img_shape:Tuple[int, int] = (1006, 590),
    seed:int = 42,
    single_channel:bool = False
    ):
    #   Model info
    #   model_name: str
    #   model: L.LightningModule
    #   data_module: L.LightningDataModule
    #   num_classes: int
    #   ckpt_file: Path
    #   img_size: Tuple[int, int]
    #   single_channel: bool
    #   facie: int (optional for experiment 2)
    #   num_points: int (optional for experiment 3)
    model_info = model_instantiator_func

    # ---- 1. Data ---- 
    data_module = get_data_module(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        img_size=None,  # Uses original image size (no resize)
        single_channel=single_channel,  # 1 or 3 channels
        batch_size=batch_size,
        seed=seed,
        num_workers=os.cpu_count()
    )

    # ---- 2. Model and wrapper ----

    # Let's check if padding is needed.
    # If shape of model and data is the same, no padding is needed
    # PS: ...SAM not need padding...
    pad_dict = None
    model_input_shape = (
        1 if single_channel else 3,
        *img_shape,
    )
    model_output_shape = (n_classes, 1, *img_shape)

    # Load model
    model = load_model_from_info(model_info)
    model = model.eval()

    # ---- 3. Return ----
    return {
        "model": model,
        "model_name": model_info["model_name"],
        "data_module": data_module,
        "ckpt_file": model_info["ckpt_file"],
        "model_input_shape": model_input_shape,
        "model_output_shape": model_output_shape,
        "pad": pad_dict,
        "single_channel": single_channel
    }

def perform_inference(
        root_data_dir,
        root_annotation_dir,
        model_instantiator_func,
        predictions_path: Path,
        batch_size:int = 1,
        n_classes:int = 6,
        img_shape: Tuple[int, int] = (1006, 590),
        num_points:int = 10,
        seed:int = 42,
        single_channel:bool = False,
        accelerator: str = "gpu",
        devices: int = 0,
        using_methodology: int = 2,
        model_idx: str = "model_name",
        evaluate_this_samples: list = [0, 199]
        ):
    model_info = load_model_and_data_module(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        model_instantiator_func=model_instantiator_func,
        batch_size=batch_size,
        n_classes=n_classes,
        img_shape=img_shape,
        seed=seed,
        single_channel=single_channel
    )
    print(f"Loading model from ckpt at: {model_info['ckpt_file']}")

    predictions_file = predictions_path / f"{model_info['model_name']}.npy"
    if predictions_file.exists():
        print(f"Predictions already exist at {predictions_file}. Skipping inference.")
        return None
    
    predictions = AUCInferencer.run(
        model=model_info["model"], 
        data_module=model_info["data_module"],
        accelerator=accelerator,
        devices=devices,
        num_points=num_points,
        using_methodology=using_methodology,
        model_idx=model_idx,
        execute_only_predictions=False,
        evaluate_this_samples=evaluate_this_samples
        )

    # TODO: batch_size not be need to be 1 in all time. 
    # at this moment, it need to be 1 because of differences 
    # of number of samples into batches: before apply torch.stack(predictions, dim=0) 
    # is need get all predictions separately.

    predictions = [p for p in predictions if p is not None]
    predictions = [p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in predictions]
    
    predictions = torch.stack(predictions, dim=0) # type: ignore
    predictions = predictions.squeeze()
    predictions = predictions.float().cpu().numpy()
    np.save(predictions_file, predictions)
    
    print(f"Predictions saved at {predictions_file}. Shape: {predictions.shape}")
    return predictions_file

def main():
    root_data_dir = "/workspaces/Minerva-Discovery/shared_data/seam_ai_datasets/seam_ai/images"
    root_annotation_dir = "/workspaces/Minerva-Discovery/shared_data/seam_ai_datasets/seam_ai/annotations"
    img_shape = (1006, 590)
    batch_size = 1 # TODO: batch_size, in this case, need to be 1 because of differences of number of samples into batches: before apply torch.stack(predictions, dim=0) is need get all predictions separately.
    seed = 42
    accelerator = "gpu"
    devices = 1
    single_channel = False
    num_points = 10
    using_methodology = 2 # 1 for use process_v1() or 2 for use process_v2()
    evaluate_this_samples = None #[0, 199]
    
    finetuned_models_path = Path.cwd() / "tmp" / "logs"
    
    for path in finetuned_models_path.iterdir():
        if path.is_dir():
            predictions_path = Path.cwd() / "tmp" / "predictions" / f"methodology_{using_methodology}" / path.name
            predictions_path.mkdir(parents=True, exist_ok=True)

            ckpt_file = f"/workspaces/Minerva-Discovery/my_experiments/sam_original/evaluate_experiments/parihaka/tmp/logs/{path.name}/seam_ai/checkpoints/last.ckpt"

            try:
                if path.name == "sam_vit_b_experiment_1":
                    model_instantiator_func = init_sam(
                        model_name=path.name,
                        ckpt_file=ckpt_file,
                        another_args={
                            "apply_freeze": {"prompt_encoder": True, "image_encoder": False, "mask_decoder": False},
                            "apply_adapter": {}
                        },
                        num_classes=6,
                        return_prediction_only=True,
                        apply_load_from_checkpoint=True
                    )
                elif path.name == "sam_vit_b_experiment_2":
                    model_instantiator_func = init_sam(
                        model_name=path.name,
                        ckpt_file=ckpt_file,
                        num_classes=3,
                        return_prediction_only=True,
                        apply_load_from_checkpoint=True
                    )
                elif path.name == "sam_vit_b_experiment_3" or path.name == "sam_vit_b_experiment_4_with_skeleton" or path.name == "sam_vit_b_experiment_4_with_percentage_of_pixels_in_grid":
                    model_instantiator_func = init_sam(
                        model_name=path.name,
                        ckpt_file=ckpt_file,
                        another_args={
                            "apply_freeze": {"image_encoder": False, "prompt_encoder": False, "mask_decoder": False},
                        },
                        num_classes=3,
                        return_prediction_only=True,
                        apply_load_from_checkpoint=True
                    )
                else:
                    print(f"Unknown model: {path.name}. Continue for next model...")
                    continue
                
                print(f"Reading Folder: {path.name}")
                model_name = model_instantiator_func['model_name']
                print("-"*80)

                print("*"*20)
                print(f"Model: {model_name}")
                print("*"*20)
                
                perform_inference(
                    root_data_dir=root_data_dir, 
                    root_annotation_dir=root_annotation_dir,
                    model_instantiator_func=model_instantiator_func, 
                    predictions_path=predictions_path,
                    batch_size=batch_size,
                    n_classes=model_instantiator_func['num_classes'],
                    img_shape=img_shape,
                    num_points=num_points,
                    seed=seed,
                    single_channel=single_channel,
                    accelerator=accelerator,
                    devices=devices,
                    using_methodology=using_methodology,
                    model_idx=path.name,
                    evaluate_this_samples=evaluate_this_samples
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error executing model: {model_name}")
            print("-"*80, "\n")

if __name__ == "__main__":
    main()