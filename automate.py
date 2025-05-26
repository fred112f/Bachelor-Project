from qgis.core import (
    QgsProject,
    QgsMapSettings,
    QgsMapRendererParallelJob
)
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.PyQt.QtCore import QSize
from qgis.utils import iface
import os
import json 
import re
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from PyQt5.QtWidgets import QApplication
import cv2
import matplotlib.pyplot as plt

script_dir = os.path.dirname(QgsApplication.qgisSettingsDirPath())
generator_folder = os.path.join(script_dir, 'generate')
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'screenshot.png')
extent_path = os.path.join(script_dir, 'screenshot_extent.json')
canvas = iface.mapCanvas()
map_settings = QgsMapSettings()
map_settings.setExtent(canvas.extent())
map_settings.setOutputSize(canvas.size())
map_settings.setLayers(canvas.layers())  
map_settings.setBackgroundColor(canvas.canvasColor())
image = QImage(map_settings.outputSize(), QImage.Format_ARGB32_Premultiplied)
image.fill(0)
painter = QPainter(image)
job = QgsMapRendererParallelJob(map_settings)
job.start()
job.waitForFinished()
job.renderedImage().save(image_path, 'png')
painter.end()
extent = canvas.extent()
extent_dict = {
    "xmin": extent.xMinimum(),
    "xmax": extent.xMaximum(),
    "ymin": extent.yMinimum(),
    "ymax": extent.yMaximum(),
    "crs": canvas.mapSettings().destinationCrs().authid()
}

with open(extent_path, 'w') as f:
    json.dump(extent_dict, f, indent=2)
print(f"Screenshot saved to: {image_path}")
print(f"Coordinate extent saved to: {extent_path}")

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.model(x)
        x = torch.cat((x, skip), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64,  normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

def load_generator(gen_path, device="cuda"):
    generator = GeneratorUNet(3, 3).to(device)
    state_dict = torch.load(gen_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator

def tile_inference_pad(generator, input_image_path, tile_size, device="cuda"):
    src = Image.open(input_image_path).convert("RGB")
    w_org, h_org = src.size
    out_final = np.zeros((h_org, w_org, 3), dtype=np.uint8)
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def run_generator_on_patch(pil_patch, real_w, real_h):
        pad_w = max(0, tile_size - real_w)
        pad_h = max(0, tile_size - real_h)
        if pad_w > 0 or pad_h > 0:
            patch_padded = Image.new("RGB", (real_w + pad_w, real_h + pad_h), (170, 211, 223))
            patch_padded.paste(pil_patch, (0, 0))
        else:
            patch_padded = pil_patch
        patch_t = to_tensor(patch_padded).unsqueeze(0).to(device)
        with torch.no_grad():
            out_t = generator(patch_t)
        out_t = (out_t * 0.5) + 0.5
        out_t = out_t.clamp(0, 1)
        out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        out_np = out_np.astype(np.uint8)
        return out_np[:real_h, :real_w, :]  
    for top in range(0, h_org, tile_size):
        for left in range(0, w_org, tile_size):
            patch_w = min(tile_size, w_org - left)
            patch_h = min(tile_size, h_org - top)
            patch_pil = src.crop((left, top, left + patch_w, top + patch_h))
            out_patch = run_generator_on_patch(patch_pil, patch_w, patch_h)
            out_final[top:top + patch_h, left:left + patch_w, :] = out_patch
    return Image.fromarray(out_final)

def process_image(input_image_path, label, generator_folder, device="cpu"):
    all_files = os.listdir(generator_folder)
    pattern = re.compile(r"^generator_epoch_(\d+)\.pth$")
    gens = []
    for f in all_files:
        m = pattern.match(f)
        if m:
            epoch_num = int(m.group(1))
            gens.append((epoch_num, os.path.join(generator_folder, f)))
    gens.sort(key=lambda x: x[0])

    if not gens:
        return None
    src = Image.open(input_image_path).convert("RGB")
    w_org, h_org = src.size
    tile_size = max(256, ((max(w_org, h_org) + 255) // 256) * 256)
    out_name = None

    for (epoch, gen_path) in gens:
        generator = load_generator(gen_path, device=device)
        fake_out = tile_inference_pad(generator, input_image_path, tile_size, device=device)
        out_name = f"{label}_epoch_{epoch}_tile_{tile_size}.png"
        fake_out.save(out_name)
    return out_name

def pixel_to_map_img1(u1, v1, width1, height1, xmin, xmax, ymin, ymax):
    if width1 <= 1 or height1 <= 1:
        raise ValueError("Invalid image dimensions.")
    X = xmin + (u1 / (width1 - 1)) * (xmax - xmin)
    Y = ymax - (v1 / (height1 - 1)) * (ymax - ymin)
    return X, Y

def img2_to_img1_pixels(u2, v2, M_inv):
    pt2_h = np.array([u2, v2, 1.0], dtype=np.float64)
    pt1_h = M_inv.dot(pt2_h)
    w = pt1_h[2]
    if abs(w) < 1e-12:
        return None, None
    u1 = pt1_h[0] / w
    v1 = pt1_h[1] / w
    return (u1, v1)

def write_points_file(
    out_path,
    img2_pixel_coords,
    map_coords,
    crs_wkt=None
):
    with open(out_path, "w", encoding="utf-8") as f:
        if crs_wkt:
            f.write(f"#CRS: {crs_wkt}\n")
        f.write("mapX,mapY,pixelX,pixelY,enable\n")
        for (px, py), (mx, my) in zip(img2_pixel_coords, map_coords):
            if mx is None or my is None:
                continue
            line = f"{mx:.6f},{my:.6f},{px:.3f},{py*-1:.3f},1\n"
            f.write(line)

def sift_ransac_homography(
    img_path1,
    img_path2,
    ratio_thresh=0.85,
    ransac_method=cv2.RANSAC,
    ransac_reproj_threshold=30.0,
    nfeatures=0,
    nOctaveLayers=40,
    contrastThreshold=0.04,
    edgeThreshold=5,
    sigma=1
):
    
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return None, None, None, None, None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in matches_knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh*n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None, kp1, kp2, None, img1, img2

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, ransac_method, ransac_reproj_threshold)
    if M is None:
        return None, kp1, kp2, None, img1, img2

    inlier_matches = []
    for i, mm in enumerate(good_matches):
        if mask[i] == 1:
            inlier_matches.append(mm)

    return M, kp1, kp2, inlier_matches, img1, img2

def main(translate_screenshot=True):
    device = "cpu"
    generator_folder = "generate"
    screenshot_path = "screenshot.png"
    if translate_screenshot and os.path.exists(screenshot_path):
        screenshot_name = process_image(screenshot_path, "screenshot", generator_folder, device=device)
    else:
        screenshot_name = screenshot_path if os.path.exists(screenshot_path) else None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_in_folder = [f for f in os.listdir(script_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    possible_user_images = [f for f in images_in_folder
                            if not f.lower().startswith('screenshot') and not f.lower().startswith('user')]
    if not possible_user_images:
        return
    user_input_original = possible_user_images[0]
    user_input_path = os.path.join(script_dir, user_input_original)
    if not os.path.exists(user_input_path):
        return
    user_input_tiled = process_image(user_input_path, "user_input", generator_folder, device=device)
    if not user_input_tiled:
        return
    if not screenshot_name or not user_input_tiled:
        return
    M, kp1, kp2, inlier_matches, img1, img2 = sift_ransac_homography(
        screenshot_name,
        user_input_tiled,
        ratio_thresh=0.85,
        ransac_method=cv2.RANSAC,
        ransac_reproj_threshold=30.0,
        nfeatures=0,
        nOctaveLayers=40,
        contrastThreshold=0.04,
        edgeThreshold=5,
        sigma=1
    )
    if M is None or img1 is None or img2 is None or not inlier_matches:
        return
    json_path = os.path.join(script_dir, "screenshot_extent.json")
    if not os.path.exists(json_path):
        return
    with open(json_path, "r") as f:
        extent_data = json.load(f)
    xmin = extent_data["xmin"]
    xmax = extent_data["xmax"]
    ymin = extent_data["ymin"]
    ymax = extent_data["ymax"]
    crs = extent_data.get("crs", "EPSG:3857")
    M_inv = np.linalg.inv(M)
    height1, width1 = img1.shape[:2]
    img2_pixels = []
    map_coords = []
    for m in inlier_matches:
        (u2, v2) = kp2[m.trainIdx].pt
        (u1_equiv, v1_equiv) = img2_to_img1_pixels(u2, v2, M_inv)
        if u1_equiv is None:
            img2_pixels.append((u2, v2))
            map_coords.append((None, None))
            continue
        X2, Y2 = pixel_to_map_img1(u1_equiv, v1_equiv, width1, height1, xmin, xmax, ymin, ymax)
        img2_pixels.append((u2, v2))
        map_coords.append((X2, Y2))

    out_points_file = os.path.join(script_dir, "georeference.points")
    write_points_file(
        out_path=out_points_file,
        img2_pixel_coords=img2_pixels,
        map_coords=map_coords,
        crs_wkt=crs
    )
    print("Done")
app = QApplication(sys.argv)
main(translate_screenshot=False)

