import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# Allow imports from the project root.
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


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

RANDOM_SEED = 42

IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.80

NUM_WORKERS = 0
THRESHOLD = 0.5

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_unet_model.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_unet_model.pth"
HISTORY_PATH = PLOT_DIR / "training_history.json"


def set_random_seed(seed: int) -> None:
    """
    Make the train-validation split and training setup reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """
    Select Apple Silicon GPU when available.
    Otherwise, fall back to CUDA or CPU.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    The model outputs logits. Sigmoid is applied inside the loss.
    """

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
    """
    Combination of BCEWithLogitsLoss and Dice loss.

    BCE provides stable pixel-level learning.
    Dice loss addresses foreground-background imbalance.
    """

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
    """
    Calculate mean Dice score and IoU for one batch.
    """

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


def create_data_loaders() -> Tuple[
    DataLoader,
    DataLoader,
]:
    """
    Create reproducible 80/20 training and validation splits.
    """

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

    train_dataset, validation_dataset = random_split(
        dataset,
        lengths=[
            train_size,
            validation_size,
        ],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("\nDataset split")
    print("-------------")
    print(f"Total pairs: {total_size}")
    print(f"Training pairs: {train_size}")
    print(
        f"Validation pairs: "
        f"{validation_size}"
    )
    print(f"Batch size: {BATCH_SIZE}")

    return train_loader, validation_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_number: int,
) -> Dict[str, float]:
    """
    Train the model for one full epoch.
    """

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress_bar = tqdm(
        loader,
        desc=(
            f"Epoch {epoch_number}/{NUM_EPOCHS} "
            f"- Training"
        ),
    )

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        loss.backward()
        optimizer.step()

        dice_score, iou_score = (
            calculate_batch_metrics(
                logits.detach(),
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
            loss=f"{running_loss / batches_completed:.4f}",
            dice=f"{running_dice / batches_completed:.4f}",
            iou=f"{running_iou / batches_completed:.4f}",
        )

    number_of_batches = len(loader)

    return {
        "loss": running_loss / number_of_batches,
        "dice": running_dice / number_of_batches,
        "iou": running_iou / number_of_batches,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_number: int,
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.
    """

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress_bar = tqdm(
        loader,
        desc=(
            f"Epoch {epoch_number}/{NUM_EPOCHS} "
            f"- Validation"
        ),
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
            loss=f"{running_loss / batches_completed:.4f}",
            dice=f"{running_dice / batches_completed:.4f}",
            iou=f"{running_iou / batches_completed:.4f}",
        )

    number_of_batches = len(loader)

    return {
        "loss": running_loss / number_of_batches,
        "dice": running_dice / number_of_batches,
        "iou": running_iou / number_of_batches,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_dice: float,
    path: Path,
) -> None:
    """
    Save model weights and training state.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "validation_loss": validation_loss,
        "validation_dice": validation_dice,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    torch.save(
        checkpoint,
        path,
    )


def save_training_history(
    history: Dict[str, List[float]],
) -> None:
    """
    Save all numerical training results as JSON.
    """

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )


def plot_training_history(
    history: Dict[str, List[float]],
) -> None:
    """
    Save loss, Dice, and IoU curves.
    """

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss",
    )
    plt.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "loss_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_dice"],
        marker="o",
        label="Training Dice",
    )
    plt.plot(
        epochs,
        history["validation_dice"],
        marker="o",
        label="Validation Dice",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "dice_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_iou"],
        marker="o",
        label="Training IoU",
    )
    plt.plot(
        epochs,
        history["validation_iou"],
        marker="o",
        label="Validation IoU",
    )
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Training and Validation IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "iou_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


@torch.no_grad()
def save_prediction_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    number_of_examples: int = 3,
) -> None:
    """
    Save original image, ground truth, predicted probability,
    and final binary prediction for validation samples.
    """

    model.eval()

    images, masks = next(
        iter(loader)
    )

    images = images.to(device)
    masks = masks.to(device)

    logits = model(images)
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= THRESHOLD
    ).float()

    examples_to_save = min(
        number_of_examples,
        images.shape[0],
    )

    for index in range(
        examples_to_save
    ):
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
            / f"validation_prediction_{index + 1}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


def main() -> None:
    set_random_seed(
        RANDOM_SEED
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("\nTraining configuration")
    print("----------------------")
    print(f"Device: {device}")
    print(
        f"Image size: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    train_loader, validation_loader = (
        create_data_loaders()
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    criterion = CombinedLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "train_dice": [],
        "validation_dice": [],
        "train_iou": [],
        "validation_iou": [],
    }

    best_validation_loss = float("inf")
    best_epoch = 0

    training_start_time = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_number=epoch,
        )

        validation_metrics = (
            validate_one_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                epoch_number=epoch,
            )
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )
        history["validation_loss"].append(
            validation_metrics["loss"]
        )
        history["train_dice"].append(
            train_metrics["dice"]
        )
        history["validation_dice"].append(
            validation_metrics["dice"]
        )
        history["train_iou"].append(
            train_metrics["iou"]
        )
        history["validation_iou"].append(
            validation_metrics["iou"]
        )

        epoch_duration = (
            time.time() - epoch_start_time
        )

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS} summary"
        )
        print(
            f"Train loss: "
            f"{train_metrics['loss']:.4f}"
        )
        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f}"
        )
        print(
            f"Train Dice: "
            f"{train_metrics['dice']:.4f}"
        )
        print(
            f"Validation Dice: "
            f"{validation_metrics['dice']:.4f}"
        )
        print(
            f"Train IoU: "
            f"{train_metrics['iou']:.4f}"
        )
        print(
            f"Validation IoU: "
            f"{validation_metrics['iou']:.4f}"
        )
        print(
            f"Epoch duration: "
            f"{epoch_duration / 60:.2f} minutes"
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=(
                    validation_metrics["loss"]
                ),
                validation_dice=(
                    validation_metrics["dice"]
                ),
                path=BEST_MODEL_PATH,
            )

            print(
                f"Best model updated at "
                f"epoch {epoch}."
            )

        save_training_history(
            history
        )

        plot_training_history(
            history
        )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=NUM_EPOCHS,
        validation_loss=(
            history["validation_loss"][-1]
        ),
        validation_dice=(
            history["validation_dice"][-1]
        ),
        path=FINAL_MODEL_PATH,
    )

    best_checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    save_prediction_examples(
        model=model,
        loader=validation_loader,
        device=device,
        number_of_examples=3,
    )

    total_duration = (
        time.time() - training_start_time
    )

    print("\nTraining complete")
    print("-----------------")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )
    print(
        f"Best validation Dice: "
        f"{best_checkpoint['validation_dice']:.4f}"
    )
    print(
        f"Total training time: "
        f"{total_duration / 60:.2f} minutes"
    )
    print(
        f"Best model saved to: "
        f"{BEST_MODEL_PATH}"
    )
    print(
        f"Final model saved to: "
        f"{FINAL_MODEL_PATH}"
    )
    print(
        f"Plots saved to: "
        f"{PLOT_DIR}"
    )
    print(
        f"Predictions saved to: "
        f"{PREDICTION_DIR}"
    )


if __name__ == "__main__":
    main()