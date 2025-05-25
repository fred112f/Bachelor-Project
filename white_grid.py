import os
import shutil
from PIL import Image
import multiprocessing
from tqdm import tqdm

source_dir = "screenshots_SE"
dest_dir = "screenshots_SE_Gray"
os.makedirs(dest_dir, exist_ok=True)

def is_gray_204(rgba):
    r, g, b, a = rgba
    return (r == 204 and g == 204 and b == 204 and a == 255)

def image_contains_5x5_gray_204(png_path):
    with Image.open(png_path) as img:
        rgba_img = img.convert("RGBA")
        width, height = rgba_img.size
        px = rgba_img.load()
        for y in range(height - 4):
            for x in range(width - 4):
                all_gray_204 = True
                for yy in range(y, y + 5):
                    for xx in range(x, x + 5):
                        if not is_gray_204(px[xx, yy]):
                            all_gray_204 = False
                            break
                    if not all_gray_204:
                        break
                if all_gray_204:
                    return True
    return False

def main():
    png_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".png")]
    full_paths = [os.path.join(source_dir, f) for f in png_files]
    results = []
    with multiprocessing.Pool(processes=15) as pool:
        for i, res in enumerate(tqdm(pool.imap(image_contains_5x5_gray_204, full_paths),
                                     total=len(full_paths),
                                     desc="Checking images")):
            results.append(res)
    for path, has_5x5_gray_204 in zip(full_paths, results):
        if has_5x5_gray_204:
            filename = os.path.basename(path)
            shutil.copy2(path, os.path.join(dest_dir, filename))

if __name__ == "__main__":
    main()
