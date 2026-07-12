from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROCESSED_IMAGE_DIR = Path("data/processed/images")
PROCESSED_MASK_DIR = Path("data/processed/masks")
OUTPUT_DIR = Path("outputs/preprocessing")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SkinLesionDataset(Dataset):
    """
    PyTorch dataset for skin lesion segmentation.

    Each processed dermoscopy image is paired with its corresponding
    binary ground-truth segmentation mask.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory does not exist: {self.mask_dir}"
            )

        image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise RuntimeError(
                f"No processed images were found in: {self.image_dir}"
            )

        self.valid_pairs = []

        for image_path in image_paths:
            image_id = image_path.stem

            mask_path = (
                self.mask_dir
                / f"{image_id}_segmentation.png"
            )

            if mask_path.exists():
                self.valid_pairs.append(
                    (image_path, mask_path)
                )

        if not self.valid_pairs:
            raise RuntimeError(
                "No valid image-mask pairs were found."
            )

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.mask_transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.valid_pairs[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)

        mask_tensor = (
            mask_tensor > 0.5
        ).float()

        return image_tensor, mask_tensor


def denormalize_image(
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image_tensor * std + mean

    return torch.clamp(
        image,
        0.0,
        1.0,
    )


def test_single_sample(
    dataset: SkinLesionDataset,
) -> None:
    """
    Load one image-mask pair and verify its shape,
    data type, value range, and visual alignment.
    """

    image_tensor, mask_tensor = dataset[0]

    print("\nSingle sample test")
    print("------------------")
    print(f"Total valid pairs: {len(dataset)}")
    print(
        f"Image tensor shape: "
        f"{tuple(image_tensor.shape)}"
    )
    print(
        f"Mask tensor shape: "
        f"{tuple(mask_tensor.shape)}"
    )
    print(
        f"Image data type: "
        f"{image_tensor.dtype}"
    )
    print(
        f"Mask data type: "
        f"{mask_tensor.dtype}"
    )
    print(
        f"Image value range: "
        f"{image_tensor.min().item():.4f} "
        f"to {image_tensor.max().item():.4f}"
    )
    print(
        f"Mask unique values: "
        f"{torch.unique(mask_tensor).tolist()}"
    )

    assert image_tensor.shape == (
        3,
        512,
        512,
    ), "Expected image shape (3, 512, 512)."

    assert mask_tensor.shape == (
        1,
        512,
        512,
    ), "Expected mask shape (1, 512, 512)."

    unique_mask_values = torch.unique(
        mask_tensor
    ).tolist()

    assert all(
        value in [0.0, 1.0]
        for value in unique_mask_values
    ), "Mask must contain only 0 and 1."

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    display_image = denormalize_image(
        image_tensor
    )

    display_image = (
        display_image
        .permute(1, 2, 0)
        .numpy()
    )

    display_mask = (
        mask_tensor
        .squeeze(0)
        .numpy()
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(display_image)
    plt.title("Loaded Processed Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(
        display_mask,
        cmap="gray",
    )
    plt.title("Loaded Binary Mask")
    plt.axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "dataset_loader_test.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Visualization saved to: "
        f"{output_path}"
    )
    print(
        "Single sample test completed successfully."
    )


def test_dataloader(
    dataset: SkinLesionDataset,
) -> None:
    """
    Verify that multiple image-mask pairs can be loaded
    together as one training batch.
    """

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    print("\nDataLoader batch test")
    print("---------------------")
    print(
        f"Batch image shape: "
        f"{tuple(images.shape)}"
    )
    print(
        f"Batch mask shape: "
        f"{tuple(masks.shape)}"
    )
    print(
        f"Batch image data type: "
        f"{images.dtype}"
    )
    print(
        f"Batch mask data type: "
        f"{masks.dtype}"
    )
    print(
        f"Batch mask values: "
        f"{torch.unique(masks).tolist()}"
    )

    assert images.shape == (
        4,
        3,
        512,
        512,
    ), (
        "Expected batch image shape "
        "(4, 3, 512, 512)."
    )

    assert masks.shape == (
        4,
        1,
        512,
        512,
    ), (
        "Expected batch mask shape "
        "(4, 1, 512, 512)."
    )

    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32

    unique_mask_values = torch.unique(
        masks
    ).tolist()

    assert all(
        value in [0.0, 1.0]
        for value in unique_mask_values
    ), "Batch masks must contain only 0 and 1."

    print(
        "DataLoader batch test completed successfully."
    )


def main() -> None:
    dataset = SkinLesionDataset(
        image_dir=PROCESSED_IMAGE_DIR,
        mask_dir=PROCESSED_MASK_DIR,
    )

    test_single_sample(dataset)
    test_dataloader(dataset)

    print(
        "\nAll dataset and DataLoader tests passed."
    )


if __name__ == "__main__":
    main()