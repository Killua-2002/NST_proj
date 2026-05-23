from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# =========================
# CONFIG
# =========================

INPUT_DIR = Path("source_data/overlap_raw")

BEST_MODEL_PATH = Path("results/best_unet.keras")
FINAL_MODEL_PATH = Path("results/final_unet.keras")

OUTPUT_DIR = Path("results/real_predictions")
OUT_IMAGES = OUTPUT_DIR / "processed_images"
OUT_MASK_A = OUTPUT_DIR / "masks_A"
OUT_MASK_B = OUTPUT_DIR / "masks_B"
OUT_MASK_C = OUTPUT_DIR / "masks_C"
OUT_CONTOURS = OUTPUT_DIR / "contours"
OUT_OVERLAYS = OUTPUT_DIR / "overlays"
OUT_VISUALS = OUTPUT_DIR / "visualizations"

IMG_SIZE = 256

THRESH_A = 0.5
THRESH_B = 0.5
THRESH_C = 0.4

for folder in [
    OUT_IMAGES,
    OUT_MASK_A,
    OUT_MASK_B,
    OUT_MASK_C,
    OUT_CONTOURS,
    OUT_OVERLAYS,
    OUT_VISUALS,
]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# PREPROCESS
# =========================

def resize_with_padding_image(img, target_size=256):
    """
    Convert ảnh về grayscale, resize giữ tỉ lệ,
    padding nền trắng để thành target_size x target_size.
    """
    img = img.convert("L")

    w, h = img.size
    scale = min(target_size / w, target_size / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (target_size, target_size), 255)

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas


def prepare_input(img):
    """
    PIL image -> tensor input cho model
    shape = 1 x 256 x 256 x 1
    """
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


# =========================
# POSTPROCESS
# =========================

def save_binary_mask(mask, path):
    """
    mask bool -> image 0/255
    """
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)


def clean_mask(mask):
    """
    Loại noise nhỏ bằng morphology.
    """
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned.astype(bool)


def mask_to_contour(mask):
    """
    Tạo contour từ mask.
    """
    mask_uint8 = (mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_img = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=1)

    return contour_img


def make_overlay(base_img, mask_A, mask_B, mask_C):
    """
    Overlay:
    A = đỏ
    B = xanh lá
    C = vàng
    """
    base = np.array(base_img.convert("RGB")).astype(np.float32)
    overlay = base.copy()

    A = mask_A.astype(bool)
    B = mask_B.astype(bool)
    C = mask_C.astype(bool)

    overlay[A] = overlay[A] * 0.4 + np.array([255, 0, 0]) * 0.6
    overlay[B] = overlay[B] * 0.4 + np.array([0, 255, 0]) * 0.6
    overlay[C] = overlay[C] * 0.3 + np.array([255, 255, 0]) * 0.7

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def make_contour_preview(base_img, mask_A, mask_B, mask_C):
    """
    Ảnh contour:
    A = đỏ
    B = xanh lá
    C = vàng
    """
    base = np.array(base_img.convert("RGB")).copy()

    contour_A = mask_to_contour(mask_A) > 0
    contour_B = mask_to_contour(mask_B) > 0
    contour_C = mask_to_contour(mask_C) > 0

    base[contour_A] = [255, 0, 0]
    base[contour_B] = [0, 255, 0]
    base[contour_C] = [255, 255, 0]

    return Image.fromarray(base)


def to_rgb(img):
    """
    Convert ảnh grayscale hoặc mask sang RGB để ghép panel.
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.astype(np.uint8))
    return img.convert("RGB")


def add_title(img, title, bar_height=28):
    """
    Thêm tiêu đề lên trên mỗi panel.
    """
    img = img.convert("RGB")
    w, h = img.size

    canvas = Image.new("RGB", (w, h + bar_height), (255, 255, 255))
    canvas.paste(img, (0, bar_height))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, bar_height], fill=(230, 230, 230))
    draw.text((8, 6), title, fill=(0, 0, 0))

    return canvas


def make_legend(draw, x, y):
    """
    Vẽ legend A/B/C.
    """
    box = 16
    gap = 8
    line_h = 24

    legend_items = [
        ((255, 0, 0), "A = Chromosome A"),
        ((0, 255, 0), "B = Chromosome B"),
        ((255, 255, 0), "C = Overlap"),
    ]

    draw.rectangle([x - 8, y - 8, x + 220, y + 78], fill=(255, 255, 255), outline=(0, 0, 0))

    for i, (color, text) in enumerate(legend_items):
        yy = y + i * line_h
        draw.rectangle([x, yy, x + box, yy + box], fill=color, outline=(0, 0, 0))
        draw.text((x + box + gap, yy), text, fill=(0, 0, 0))


def make_visualization(base_img, overlay_img, contour_img, mask_A, mask_B, mask_C, save_path):
    """
    Tạo ảnh tổng hợp để nhìn trực quan.
    Gồm:
    - processed input
    - overlay
    - contour
    - mask A
    - mask B
    - mask C
    + legend A/B/C
    """
    mask_A_img = Image.fromarray((mask_A.astype(np.uint8) * 255), mode="L")
    mask_B_img = Image.fromarray((mask_B.astype(np.uint8) * 255), mode="L")
    mask_C_img = Image.fromarray((mask_C.astype(np.uint8) * 255), mode="L")

    panels = [
        add_title(to_rgb(base_img), "Input"),
        add_title(to_rgb(overlay_img), "Overlay"),
        add_title(to_rgb(contour_img), "Contour"),
        add_title(to_rgb(mask_A_img), "Mask A"),
        add_title(to_rgb(mask_B_img), "Mask B"),
        add_title(to_rgb(mask_C_img), "Mask C"),
    ]

    panel_w, panel_h = panels[0].size
    cols = 3
    rows = 2
    margin = 10
    legend_h = 100

    canvas_w = cols * panel_w + (cols + 1) * margin
    canvas_h = rows * panel_h + (rows + 1) * margin + legend_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = margin + c * (panel_w + margin)
            y = margin + r * (panel_h + margin)
            canvas.paste(panels[idx], (x, y))
            idx += 1

    draw = ImageDraw.Draw(canvas)
    make_legend(draw, 20, rows * panel_h + (rows + 1) * margin + 10)

    canvas.save(save_path)


def get_image_paths(input_dir):
    """
    Hỗ trợ png/jpg/jpeg.
    """
    image_paths = []
    patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]

    for pattern in patterns:
        image_paths.extend(input_dir.glob(pattern))

    return sorted(image_paths)


# =========================
# MAIN
# =========================

def main():
    if FINAL_MODEL_PATH.exists():
        model_path = FINAL_MODEL_PATH
    elif BEST_MODEL_PATH.exists():
        model_path = BEST_MODEL_PATH
    else:
        raise FileNotFoundError(
            "No model found. Expected results/final_unet.keras or results/best_unet.keras"
        )

    image_paths = get_image_paths(INPUT_DIR)

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No image files found in {INPUT_DIR}. Supported: png, jpg, jpeg."
        )

    print(f"Found {len(image_paths)} real overlap images.")
    print(f"Loading model: {model_path}")

    model = keras.models.load_model(model_path, compile=False)

    for idx, img_path in enumerate(image_paths, start=1):
        name = img_path.stem + ".png"

        img = Image.open(img_path)
        processed_img = resize_with_padding_image(img, IMG_SIZE)

        x = prepare_input(processed_img)
        pred = model.predict(x, verbose=0)[0]

        prob_A = pred[:, :, 0]
        prob_B = pred[:, :, 1]
        prob_C = pred[:, :, 2]

        mask_A = clean_mask(prob_A > THRESH_A)
        mask_B = clean_mask(prob_B > THRESH_B)
        mask_C = clean_mask(prob_C > THRESH_C)

        # Save processed input
        processed_img.save(OUT_IMAGES / name)

        # Save masks
        save_binary_mask(mask_A, OUT_MASK_A / name)
        save_binary_mask(mask_B, OUT_MASK_B / name)
        save_binary_mask(mask_C, OUT_MASK_C / name)

        # Save overlay
        overlay = make_overlay(processed_img, mask_A, mask_B, mask_C)
        overlay.save(OUT_OVERLAYS / name)

        # Save contour
        contour_preview = make_contour_preview(processed_img, mask_A, mask_B, mask_C)
        contour_preview.save(OUT_CONTOURS / name)

        # Save full visualization
        make_visualization(
            base_img=processed_img,
            overlay_img=overlay,
            contour_img=contour_preview,
            mask_A=mask_A,
            mask_B=mask_B,
            mask_C=mask_C,
            save_path=OUT_VISUALS / name
        )

        if idx % 20 == 0 or idx == len(image_paths):
            print(f"Predicted {idx}/{len(image_paths)}")

    print("Done real inference.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()