from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_DIR = Path("data/raw/images")
MASK_DIR = Path("data/raw/masks")

PROCESSED_IMAGE_DIR = Path("data/processed/images")
PROCESSED_MASK_DIR = Path("data/processed/masks")

TARGET_SIZE = (512, 512)


def remove_hair(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17),
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel,
    )

    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY,
    )

    repaired = cv2.inpaint(
        image,
        hair_mask,
        1,
        cv2.INPAINT_TELEA,
    )

    return repaired


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge(
        (enhanced_lightness, channel_a, channel_b)
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        image,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    hair_removed = remove_hair(resized)
    enhanced = apply_clahe(hair_removed)

    denoised = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0,
    )

    return denoised


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        mask,
        TARGET_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )

    _, binary_mask = cv2.threshold(
        resized,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return binary_mask


def main() -> None:
    PROCESSED_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    processed_count = 0
    skipped_count = 0

    for image_path in tqdm(
        image_paths,
        desc="Processing dataset",
    ):
        image_id = image_path.stem

        mask_path = (
            MASK_DIR
            / f"{image_id}_segmentation.png"
        )

        if not mask_path.exists():
            print(f"Skipping missing mask: {image_id}")
            skipped_count += 1
            continue

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            print(f"Could not read image: {image_path}")
            skipped_count += 1
            continue

        if mask is None:
            print(f"Could not read mask: {mask_path}")
            skipped_count += 1
            continue

        processed_image = preprocess_image(image)
        processed_mask = preprocess_mask(mask)

        output_image_path = (
            PROCESSED_IMAGE_DIR
            / f"{image_id}.jpg"
        )

        output_mask_path = (
            PROCESSED_MASK_DIR
            / f"{image_id}_segmentation.png"
        )

        cv2.imwrite(
            str(output_image_path),
            processed_image,
        )

        cv2.imwrite(
            str(output_mask_path),
            processed_mask,
        )

        processed_count += 1

    print("\nPreprocessing complete")
    print(f"Processed pairs: {processed_count}")
    print(f"Skipped pairs: {skipped_count}")
    print(
        f"Processed images saved to: "
        f"{PROCESSED_IMAGE_DIR}"
    )
    print(
        f"Processed masks saved to: "
        f"{PROCESSED_MASK_DIR}"
    )


if __name__ == "__main__":
    main()