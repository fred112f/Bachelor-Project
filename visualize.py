import os
import torch
import torch.nn as nn
from torchsummary import summary                       #
from torchinfo import summary as summary_info           
from torchviz import make_dot                           

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.down1 = UNetDown(in_channels,  64, normalize=False)   
        self.down2 = UNetDown(64, 128)                             
        self.down3 = UNetDown(128, 256)                            
        self.down4 = UNetDown(256, 512, dropout=0.5)              
        self.down5 = UNetDown(512, 512, dropout=0.5)               
        self.down6 = UNetDown(512, 512, dropout=0.5)              
        self.down7 = UNetDown(512, 512, dropout=0.5)               
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) 
        self.up1  = UNetUp(512, 512, dropout=0.5)   
        self.up2  = UNetUp(1024, 512, dropout=0.5)  
        self.up3  = UNetUp(1024, 512, dropout=0.5)  
        self.up4  = UNetUp(1024, 512, dropout=0.5) 
        self.up5  = UNetUp(1024, 256)               
        self.up6  = UNetUp(512,  128)               
        self.up7  = UNetUp(256,   64)             
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x);  d2 = self.down2(d1);  d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4);  d6 = self.down6(d5)
        d7 = self.down7(d6); d8 = self.down8(d7)
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
        super().__init__()
        def disc_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *disc_block(in_channels * 2,  64, normalize=False),  
            *disc_block(64, 128),                               
            *disc_block(128, 256),                               
            nn.ZeroPad2d((1, 0, 1, 0)),                         
            nn.Conv2d(256, 512, 4, padding=1),                   
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512,   1, 4, padding=1)                    
        )
    def forward(self, img_A, img_B):
        x = torch.cat((img_A, img_B), 1)
        return self.model(x)

def main():
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_SIZE  = (3, 256, 256)  
    BATCH_SIZE  = 1
    gen  = GeneratorUNet().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    summary(gen,  input_size=INPUT_SIZE)
    summary(disc, input_size=[INPUT_SIZE, INPUT_SIZE])
    summary_info(gen,  input_size=(BATCH_SIZE, *INPUT_SIZE),
                 depth=3, col_names=("output_size", "num_params"))
    summary_info(disc, input_size=[(BATCH_SIZE, *INPUT_SIZE),
                                   (BATCH_SIZE, *INPUT_SIZE)],
                 depth=3, col_names=("output_size", "num_params"))
    gen.eval();  disc.eval()                   
    os.makedirs("plots", exist_ok=True)     
    dummy_gen_in   = torch.randn(BATCH_SIZE, *INPUT_SIZE, device=DEVICE)
    gen_out        = gen(dummy_gen_in)
    gen_graph = make_dot(gen_out,
                         params=dict(gen.named_parameters()),
                         show_attrs=True, show_saved=True)
    gen_graph.render("plots/generator_model_graph", format="png", cleanup=True)
    dummy_A = torch.randn(BATCH_SIZE, *INPUT_SIZE, device=DEVICE)
    dummy_B = torch.randn(BATCH_SIZE, *INPUT_SIZE, device=DEVICE)
    disc_out = disc(dummy_A, dummy_B)
    disc_graph = make_dot(disc_out,
                          params=dict(disc.named_parameters()),
                          show_attrs=True, show_saved=True)
    disc_graph.render("plots/discriminator_model_graph", format="png", cleanup=True)
    
if __name__ == "__main__":
    main()
