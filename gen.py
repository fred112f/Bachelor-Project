import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T

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

def tile_inference_pad(
    generator,
    input_image_path,
    tile_size,
    device="cuda"
):
    src = Image.open(input_image_path).convert("RGB")
    w_org, h_org = src.size
    out_final = np.zeros((h_org, w_org, 3), dtype=np.uint8)
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    def run_generator_on_patch(pil_patch, real_w, real_h):
        pad_w = tile_size - real_w
        pad_h = tile_size - real_h
        if pad_w < 0: pad_w = 0
        if pad_h < 0: pad_h = 0
        if pad_w > 0 or pad_h > 0:
            patch_padded = Image.new("RGB", (real_w+pad_w, real_h+pad_h), (170, 211, 223))
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
        if pad_w > 0 or pad_h > 0:
            out_np = out_np[:real_h, :real_w, :]
        return out_np
    for top in range(0, h_org, tile_size):
        for left in range(0, w_org, tile_size):
            patch_w = min(tile_size, w_org - left)
            patch_h = min(tile_size, h_org - top)
            patch_pil = src.crop((left, top, left+patch_w, top+patch_h))
            out_patch = run_generator_on_patch(patch_pil, patch_w, patch_h)
            out_final[top:top+patch_h, left:left+patch_w, :] = out_patch
    return Image.fromarray(out_final)

def main():
    device = "cuda"
    generator_folder = "test"
    input_image_path = "raw_images/dagupan.png"
    all_files = os.listdir(generator_folder)
    pattern = re.compile(r"^generator_epoch_(\d+)\.pth$")
    gens = []
    for f in all_files:
        m = pattern.match(f)
        if m:
            ep = int(m.group(1))
            gens.append((ep, os.path.join(generator_folder, f)))
    gens.sort(key=lambda x: x[0])
    if not gens:
        print("No generator_epoch_XX.pth found.")
        return
    src = Image.open(input_image_path).convert("RGB")
    w_org, h_org = src.size
    tile_size = max(w_org, h_org)
    if tile_size < 256:
        tile_size = 256
    if tile_size % 256 != 0:
        tile_size = ((tile_size // 256) + 1) * 256
    for (epoch, gen_path) in gens:
        generator = load_generator(gen_path, device=device)
        fake_out = tile_inference_pad(generator, input_image_path, tile_size, device=device)
        out_name = f"epoch_{epoch}_tile_{tile_size}.png"
        fake_out.save(out_name)
        print(f"  => saved {out_name}")

if __name__ == "__main__":
    main()
