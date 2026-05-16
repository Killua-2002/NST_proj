from pathlib import Path
from PIL import Image
import numpy as np
import os
import shutil
import multiprocessing

# =========================
# CONFIG
# =========================
INPUT_IMAGE_DIR = Path("generated_data/images")
INPUT_MASK_A_DIR = Path("generated_data/masks_A")
INPUT_MASK_B_DIR = Path("generated_data/masks_B")
INPUT_MASK_C_DIR = Path("generated_data/masks_C")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", max(1, multiprocessing.cpu_count() - 1)))

OUTPUT_ROOT = Path("processed_data_256")
OUTPUT_IMAGE_DIR = OUTPUT_ROOT / "images"
OUTPUT_MASK_A_DIR = OUTPUT_ROOT / "masks_A"
OUTPUT_MASK_B_DIR = OUTPUT_ROOT / "masks_B"
OUTPUT_MASK_C_DIR = OUTPUT_ROOT / "masks_C"

TARGET_SIZE = 256
CLEAR_OLD_OUTPUT = True

if CLEAR_OLD_OUTPUT and OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for folder in [
    OUTPUT_IMAGE_DIR,
    OUTPUT_MASK_A_DIR,
    OUTPUT_MASK_B_DIR,
    OUTPUT_MASK_C_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================
# HELPER
# =========================
def resize_with_padding(img, target_size=256, is_mask=False):
    """
    Resize giữ tỉ lệ, sau đó padding để thành target_size x target_size.
    image:
        - resize: bilinear
        - padding: 255
    mask:
        - resize: nearest
        - padding: 0
    """
    w, h = img.size

    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if is_mask:
        img = img.resize((new_w, new_h), Image.NEAREST)
        canvas = Image.new("L", (target_size, target_size), 0)
    else:
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (target_size, target_size), 255)

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    canvas.paste(img, (paste_x, paste_y))
    return canvas


def binarize_mask(mask_img):
    """
    Đảm bảo mask chỉ còn 0 và 1 rồi lưu ra 0 hoặc 255.
    """
    arr = np.array(mask_img)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def process_single_sample(img_path_str):
    img_path = Path(img_path_str)
    name = img_path.name

    mask_A_path = INPUT_MASK_A_DIR / name
    mask_B_path = INPUT_MASK_B_DIR / name
    mask_C_path = INPUT_MASK_C_DIR / name

    if not (mask_A_path.exists() and mask_B_path.exists() and mask_C_path.exists()):
        return False

    img = Image.open(img_path).convert("L")
    img = resize_with_padding(img, target_size=TARGET_SIZE, is_mask=False)
    img.save(OUTPUT_IMAGE_DIR / name)

    mask_A = Image.open(mask_A_path).convert("L")
    mask_A = binarize_mask(mask_A)
    mask_A = resize_with_padding(mask_A, target_size=TARGET_SIZE, is_mask=True)
    mask_A = binarize_mask(mask_A)
    mask_A.save(OUTPUT_MASK_A_DIR / name)

    mask_B = Image.open(mask_B_path).convert("L")
    mask_B = binarize_mask(mask_B)
    mask_B = resize_with_padding(mask_B, target_size=TARGET_SIZE, is_mask=True)
    mask_B = binarize_mask(mask_B)
    mask_B.save(OUTPUT_MASK_B_DIR / name)

    mask_C = Image.open(mask_C_path).convert("L")
    mask_C = binarize_mask(mask_C)
    mask_C = resize_with_padding(mask_C, target_size=TARGET_SIZE, is_mask=True)
    mask_C = binarize_mask(mask_C)
    mask_C.save(OUTPUT_MASK_C_DIR / name)

    return True


# =========================
# PROCESS
# =========================

def main():
    image_paths = sorted(INPUT_IMAGE_DIR.glob("*.png"))

    print(f"Found {len(image_paths)} generated images.")

    if len(image_paths) == 0:
        raise FileNotFoundError("No generated images found. Run 3v1_generate_synthetic_masks.py first.")

    image_path_strs = [str(p) for p in image_paths]
    processed = 0

    if NUM_WORKERS > 1:
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            for success in pool.imap_unordered(process_single_sample, image_path_strs, chunksize=16):
                if success:
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{len(image_paths)}")
    else:
        for img_path in image_paths:
            if process_single_sample(str(img_path)):
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(image_paths)}")

    print(f"Done preprocessing to 256x256. Processed {processed} files.")


if __name__ == "__main__":
    main()
