import os
import shutil
from PIL import Image

source_dir = "screenshots_SE"
dest_dir = "screenshots_SE_Gray"
os.makedirs(dest_dir, exist_ok=True)

def image_contains_white(png_path):
    with Image.open(png_path) as img:
        rgba_img = img.convert("RGBA")
        pixels = rgba_img.getdata()
        if (204,204,204,204) in pixels:
            return True
    return False

def main():
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".png"):
            full_path = os.path.join(source_dir, filename)
            if image_contains_white(full_path):
                shutil.copy2(full_path, os.path.join(dest_dir, filename))
                
if __name__ == "__main__":
    main()
