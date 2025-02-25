from pathlib import Path
from typing import Dict, Any, Optional
from torchmetrics import JaccardIndex
from minerva.models.nets.image.sam import Sam

def init_sam(
        model_name: str,
        ckpt_file: str,
        another_args: Optional[Dict[str, Any]] = None,  # Pode ser None
        num_classes: int = 6,
        vit_model: str = 'vit-b',
        multimask_output: bool = True,
        return_prediction_only: bool = False,
        apply_load_from_checkpoint: bool = False
) -> Dict[str, Any]:
    # Garante que another_args seja um dicionário, mesmo que None tenha sido passado
    another_args = another_args or {}

    if not apply_load_from_checkpoint:
        model = Sam(
            train_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
            val_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
            test_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
            vit_type=vit_model,
            checkpoint=ckpt_file,
            num_multimask_outputs=num_classes,
            iou_head_depth=num_classes,
            multimask_output=multimask_output,
            return_prediction_only=return_prediction_only,
            **another_args  # Passa os argumentos adicionais corretamente
        )
    else:
        model = Sam.load_from_checkpoint(
            train_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
            val_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
            test_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=num_classes)},
            vit_type=vit_model,
            checkpoint_path=ckpt_file,
            num_multimask_outputs=num_classes,
            iou_head_depth=num_classes,
            multimask_output=multimask_output,
            return_prediction_only=return_prediction_only,
            **another_args  # Passa os argumentos adicionais corretamente
        )

    ckpt_file_path = Path.cwd() / "tmp" / "logs" / model_name / "seam_ai" / "checkpoints" / "last.ckpt"

    return {
        "model_name": model_name,
        "model": model,
        "num_classes": num_classes,
        "ckpt_file": ckpt_file_path,
    }