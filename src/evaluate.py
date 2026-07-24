import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.unet import UNet
from src.dataset import (
    PROCESSED_IMAGE_DIR,
    PROCESSED_MASK_DIR,
    SkinLesionDataset,
    denormalize_image,
)


RANDOM_SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 2
TRAIN_RATIO = 0.80
NUM_WORKERS = 0
THRESHOLD = 0.5

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "best_unet_model.pth"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)

METRICS_PATH = (
    EVALUATION_DIR
    / "evaluation_metrics.json"
)

PREDICTION_DIR = (
    EVALUATION_DIR
    / "predictions"
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1,
        )

        targets = targets.reshape(
            targets.shape[0],
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = self.bce_loss(
            logits,
            targets,
        )

        dice = self.dice_loss(
            logits,
            targets,
        )

        return bce + dice


def calculate_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = THRESHOLD,
    smooth: float = 1e-7,
) -> Tuple[float, float]:
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )

    targets = targets.reshape(
        targets.shape[0],
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum
        + target_sum
        + smooth
    )

    union = (
        prediction_sum
        + target_sum
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return (
        dice.mean().item(),
        iou.mean().item(),
    )


def create_validation_loader() -> DataLoader:
    dataset = SkinLesionDataset(
        image_dir=PROCESSED_IMAGE_DIR,
        mask_dir=PROCESSED_MASK_DIR,
    )

    total_size = len(dataset)

    train_size = int(
        total_size * TRAIN_RATIO
    )

    validation_size = (
        total_size - train_size
    )

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    _, validation_dataset = random_split(
        dataset,
        lengths=[
            train_size,
            validation_size,
        ],
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("\nEvaluation dataset")
    print("------------------")
    print(f"Total pairs: {total_size}")
    print(
        f"Validation pairs: "
        f"{validation_size}"
    )
    print(f"Batch size: {BATCH_SIZE}")

    return validation_loader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress_bar = tqdm(
        loader,
        desc="Evaluating best model",
    )

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        dice_score, iou_score = (
            calculate_batch_metrics(
                logits,
                masks,
            )
        )

        running_loss += loss.item()
        running_dice += dice_score
        running_iou += iou_score

        batches_completed = (
            progress_bar.n + 1
        )

        progress_bar.set_postfix(
            loss=(
                f"{running_loss / batches_completed:.4f}"
            ),
            dice=(
                f"{running_dice / batches_completed:.4f}"
            ),
            iou=(
                f"{running_iou / batches_completed:.4f}"
            ),
        )

    number_of_batches = len(loader)

    return {
        "validation_loss": (
            running_loss / number_of_batches
        ),
        "validation_dice": (
            running_dice / number_of_batches
        ),
        "validation_iou": (
            running_iou / number_of_batches
        ),
    }


@torch.no_grad()
def save_prediction_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    number_of_examples: int = 6,
) -> None:
    model.eval()

    saved_examples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        for index in range(images.shape[0]):
            if saved_examples >= number_of_examples:
                return

            display_image = denormalize_image(
                images[index].cpu()
            )

            display_image = (
                display_image
                .permute(1, 2, 0)
                .numpy()
            )

            ground_truth = (
                masks[index]
                .squeeze(0)
                .cpu()
                .numpy()
            )

            probability_mask = (
                probabilities[index]
                .squeeze(0)
                .cpu()
                .numpy()
            )

            predicted_mask = (
                predictions[index]
                .squeeze(0)
                .cpu()
                .numpy()
            )

            sample_dice, sample_iou = (
                calculate_batch_metrics(
                    logits[index:index + 1],
                    masks[index:index + 1],
                )
            )

            plt.figure(figsize=(14, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(display_image)
            plt.title("Processed Image")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(
                ground_truth,
                cmap="gray",
            )
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(
                probability_mask,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            plt.title("Probability Mask")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(
                predicted_mask,
                cmap="gray",
            )
            plt.title(
                f"Prediction\n"
                f"Dice: {sample_dice:.3f}, "
                f"IoU: {sample_iou:.3f}"
            )
            plt.axis("off")

            plt.tight_layout()

            output_path = (
                PREDICTION_DIR
                / (
                    f"evaluation_prediction_"
                    f"{saved_examples + 1}.png"
                )
            )

            plt.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight",
            )

            plt.close()

            saved_examples += 1


def main() -> None:
    set_random_seed(
        RANDOM_SEED
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("\nEvaluation configuration")
    print("------------------------")
    print(f"Device: {device}")
    print(
        f"Image size: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    validation_loader = (
        create_validation_loader()
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = CombinedLoss()

    metrics = evaluate_model(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    results = {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_loss": (
            checkpoint["validation_loss"]
        ),
        "checkpoint_validation_dice": (
            checkpoint["validation_dice"]
        ),
        "evaluation_validation_loss": (
            metrics["validation_loss"]
        ),
        "evaluation_validation_dice": (
            metrics["validation_dice"]
        ),
        "evaluation_validation_iou": (
            metrics["validation_iou"]
        ),
        "validation_samples": (
            len(validation_loader.dataset)
        ),
        "threshold": THRESHOLD,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    save_prediction_examples(
        model=model,
        loader=validation_loader,
        device=device,
        number_of_examples=6,
    )

    print("\nEvaluation complete")
    print("-------------------")
    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Validation loss: "
        f"{metrics['validation_loss']:.4f}"
    )
    print(
        f"Validation Dice: "
        f"{metrics['validation_dice']:.4f}"
    )
    print(
        f"Validation IoU: "
        f"{metrics['validation_iou']:.4f}"
    )
    print(
        f"Metrics saved to: "
        f"{METRICS_PATH}"
    )
    print(
        f"Predictions saved to: "
        f"{PREDICTION_DIR}"
    )


if __name__ == "__main__":
    main()