from pathlib import Path
import shutil

SOURCE_DIR = Path("source_data/single_chromosomes_raw")
OUTPUT_DIR = Path("source_data/single_chromosomes")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

png_paths = sorted(SOURCE_DIR.rglob("*.png"))

print(f"Found {len(png_paths)} PNG images.")

for idx, img_path in enumerate(png_paths, start=1):
    new_name = f"chr_{idx:06d}.png"
    output_path = OUTPUT_DIR / new_name

    shutil.copy2(img_path, output_path)

print(f"Copied {len(png_paths)} PNG images to {OUTPUT_DIR}")