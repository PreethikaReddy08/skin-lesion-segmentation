from pathlib import Path


IMAGE_DIR = Path("data/raw/images")
MASK_DIR = Path("data/raw/masks")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_id(path: Path) -> str:
    return path.stem


def get_mask_id(path: Path) -> str:
    return path.stem.replace("_segmentation", "")


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

    image_files = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    mask_files = sorted(
        path
        for path in MASK_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    image_ids = {get_image_id(path) for path in image_files}
    mask_ids = {get_mask_id(path) for path in mask_files}

    matched_ids = sorted(image_ids & mask_ids)
    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)

    print(f"Images found: {len(image_files)}")
    print(f"Masks found: {len(mask_files)}")
    print(f"Matched pairs: {len(matched_ids)}")
    print(f"Images without masks: {len(missing_masks)}")
    print(f"Masks without images: {len(missing_images)}")

    if missing_masks:
        print("\nSample images without masks:")
        for image_id in missing_masks[:10]:
            print(f"  - {image_id}")

    if missing_images:
        print("\nSample masks without images:")
        for mask_id in missing_images[:10]:
            print(f"  - {mask_id}")


if __name__ == "__main__":
    main()