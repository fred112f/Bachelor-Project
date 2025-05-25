import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_val_losses(csv_path):
    epochs = []
    val_G = []
    val_D = []
    if not os.path.exists(csv_path):
        return epochs, val_G, val_D
    with open(csv_path, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(row["epoch"])
            g = float(row["val_G_loss"])
            d = float(row["val_D_loss"])
            epochs.append(e)
            val_G.append(g)
            val_D.append(d)
    return epochs, val_G, val_D
def check_stability(loss_list, last_n=5, threshold=0.01):
    if len(loss_list) < last_n:
        return (False, None)  
    recent_vals = loss_list[-last_n:]
    rng = max(recent_vals) - min(recent_vals)
    is_stable = (rng < threshold)
    return (is_stable, rng)
def plot_and_check_stability(csv_path, last_n=5, threshold=0.01, output_dir=None):
    epochs, val_G, val_D = load_val_losses(csv_path)
    if not epochs:
        return
    z_g = np.polyfit(epochs, val_G, 1)
    p_g = np.poly1d(z_g)
    slope_g = z_g[0]
    z_d = np.polyfit(epochs, val_D, 1)
    p_d = np.poly1d(z_d)
    slope_d = z_d[0]
    plt.figure(figsize=(8,5))
    plt.plot(epochs, val_G, marker='o', label="Val G loss")
    plt.plot(epochs, val_D, marker='s', label="Val D loss")
    plt.plot(epochs, p_g(epochs), label=f"G best fit (slope={slope_g:.4f})")
    plt.plot(epochs, p_d(epochs), label=f"D best fit (slope={slope_d:.4f})")
    plt.title(f"Validation Losses: {os.path.basename(csv_path)}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    g_stable, g_range = check_stability(val_G, last_n=last_n, threshold=threshold)
    d_stable, d_range = check_stability(val_D, last_n=last_n, threshold=threshold)
    print(f"File: {csv_path}")
    print(f"  G_loss last {last_n} range: {g_range:.4f}, stable={g_stable}")
    print(f"  D_loss last {last_n} range: {d_range:.4f}, stable={d_stable}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_plot.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to")
    plt.show()

def main():
    folder_path = "validation_stats" 
    plot_folder = os.path.join(folder_path, "plots")
    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
    ]
    if not csv_files:
        print(f"No CSV files found in folder: {folder_path}")
        return
    LAST_N = 10    
    THRESHOLD = 1.0 
    for csv_path in csv_files:
        plot_and_check_stability(
            csv_path,
            last_n=LAST_N,
            threshold=THRESHOLD,
            output_dir=plot_folder
        )

if __name__ == "__main__":
    main()
