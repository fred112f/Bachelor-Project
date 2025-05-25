import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import random
import numpy as np

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True


class MapDataset(Dataset):
    def __init__(self, root_source, root_target, transform=None):
        super().__init__()
        self.root_source = root_source
        self.root_target = root_target
        self.source_images = sorted(os.listdir(root_source))
        self.target_images = sorted(os.listdir(root_target))
        self.transform = transform
    def __len__(self):
        return len(self.source_images)
    def __getitem__(self, idx):
        source_path = os.path.join(self.root_source, self.source_images[idx])
        target_path = os.path.join(self.root_target, self.target_images[idx])
        source_img = Image.open(source_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        return source_img, target_img

def get_dataloaders(
    root_dataset="split_dataset",
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    full_size=True
):
    if full_size:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
        ])
    else:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
        ])
    train_dataset_source = os.path.join(root_dataset, "train", "A")
    train_dataset_target = os.path.join(root_dataset, "train", "B")
    val_dataset_source   = os.path.join(root_dataset, "val", "A")
    val_dataset_target   = os.path.join(root_dataset, "val", "B")
    train_dataset = MapDataset(train_dataset_source, train_dataset_target, transform=transform)
    val_dataset   = MapDataset(val_dataset_source,   val_dataset_target,   transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

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
            nn.ReLU(inplace=True)
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
        self.down1 = UNetDown(in_channels, 64, normalize=False)
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

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64,  normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)), 
            nn.Conv2d(256, 512, 4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    def forward(self, img_A, img_B):
        x = torch.cat((img_A, img_B), 1)
        return self.model(x)

def validate_pix2pix(generator, discriminator, val_loader, device, criterion_GAN, criterion_L1, lambda_l1):
    generator.eval()
    discriminator.eval()
    total_G_loss = 0.0
    total_D_loss = 0.0
    val_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for (source, target) in val_bar:
            source = source.to(device)
            target = target.to(device)
            real_pred = discriminator(source, target)
            real_label = torch.ones_like(real_pred, device=device)
            loss_D_real = criterion_GAN(real_pred, real_label)
            fake_target = generator(source)
            fake_pred = discriminator(source, fake_target)
            fake_label = torch.zeros_like(fake_pred, device=device)
            loss_D_fake = criterion_GAN(fake_pred, fake_label)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            fake_pred = discriminator(source, fake_target)
            real_label = torch.ones_like(fake_pred, device=device)
            loss_G_GAN = criterion_GAN(fake_pred, real_label)
            loss_G_L1 = criterion_L1(fake_target, target) * lambda_l1
            loss_G = loss_G_GAN + loss_G_L1
            total_D_loss += loss_D.item()
            total_G_loss += loss_G.item()
            val_bar.set_postfix({
                "D_loss": f"{loss_D.item():.4f}",
                "G_loss": f"{loss_G.item():.4f}"
            })
    avg_D_loss = total_D_loss / len(val_loader)
    avg_G_loss = total_G_loss / len(val_loader)
    return avg_G_loss, avg_D_loss

def train_pix2pix(
    generator,
    discriminator,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=2e-4,
    device="cuda",
    lambda_l1=100.0,
    patience=10 
):
    criterion_GAN = nn.MSELoss()  
    criterion_L1  = nn.L1Loss()
    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator.train()
    discriminator.train()
    best_val_G_loss = float('inf')
    no_improvement_count = 0
    for epoch in range(num_epochs):
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        total_G_loss = 0.0
        total_D_loss = 0.0
        for i, (source, target) in enumerate(train_bar):
            source = source.to(device)
            target = target.to(device)
            opt_D.zero_grad()
            real_pred = discriminator(source, target)
            real_label = torch.ones_like(real_pred, device=device)
            loss_D_real = criterion_GAN(real_pred, real_label)
            fake_target = generator(source)
            fake_pred = discriminator(source, fake_target.detach())
            fake_label = torch.zeros_like(fake_pred, device=device)
            loss_D_fake = criterion_GAN(fake_pred, fake_label)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()
            opt_G.zero_grad()
            fake_pred = discriminator(source, fake_target)
            real_label = torch.ones_like(fake_pred, device=device)
            loss_G_GAN = criterion_GAN(fake_pred, real_label)
            loss_G_L1 = criterion_L1(fake_target, target) * lambda_l1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()
            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()
            train_bar.set_postfix({"D_loss": f"{loss_D.item():.4f}", "G_loss": f"{loss_G.item():.4f}"})
        avg_G_loss = total_G_loss / len(train_loader)
        avg_D_loss = total_D_loss / len(train_loader)
        val_G_loss, val_D_loss = validate_pix2pix(generator, discriminator, val_loader,
                                                  device, criterion_GAN, criterion_L1, lambda_l1)
        if val_G_loss < best_val_G_loss:
            best_val_G_loss = val_G_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                break
    print("Training complete!")

def main():
    NUM_EPOCHS = 1000
    BATCH_SIZE = 1
    LR = 2e-4
    DEVICE = "cuda" 
    LAMBDA_L1 = 100.0
    train_loader, val_loader = get_dataloaders(
        root_dataset="split_dataset",
        batch_size=BATCH_SIZE,
        num_workers=16,
        pin_memory=False,
        full_size=False
    )
    generator = GeneratorUNet(in_channels=3, out_channels=3).to(DEVICE)
    discriminator = Discriminator(in_channels=3).to(DEVICE)
    train_pix2pix(
        generator,
        discriminator,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=DEVICE,
        lambda_l1=LAMBDA_L1,
        patience=100  
    )
    torch.save(generator.state_dict(), "generator_pix2pix.pth")
    torch.save(discriminator.state_dict(), "discriminator_pix2pix.pth")
    print("Models saved!")

if __name__ == "__main__":
    set_random_seed(42)
    main()
