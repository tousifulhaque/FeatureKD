import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
# 27:30
#A01 = 0:130
#A02 = 0:50
#A03 = 100:350
#A05 = 50:250
#A06 = 0:130
#A07 = 0:200
#45:48
#A08 = 0:200
#3:6
#A09 = 0:130

activity_code = 'A01'
# Load the ground truth window
#watch_gt = pd.read_csv('../data/smartfallmm/young/accelerometer/watch/S48A13T01.csv').values[:, -3:][125:176, :]
#watch_gt = pd.read_csv('../data/smartfallmm/young/accelerometer/watch/S45A11T04.csv').values[525:590, -3:]
#watch_gt = pd.read_csv('../data/smartfallmm/young/accelerometer/watch/S45A13T02.csv').values[:, -3:][500:560, :]
#watch_gt = pd.read_csv('../data/smartfallmm/young/accelerometer/watch/S62A10T03.csv').values[140:200, -3:]
watch_gt = pd.read_csv(f'../data/smartfallmm/young/skeleton/S62{activity_code}T04.csv').values[0:130, 27:30]

print(watch_gt.shape)
window_length = watch_gt.shape[0]
out_dir = "../data/smartfallmm_refined/old/skeleton"
os.makedirs(out_dir, exist_ok = True)
visualization_dir = f"visualizations/old/skeleton/{activity_code}"
os.makedirs(visualization_dir, exist_ok=True)
# Find all files with 'A13' in the filename
files = glob.glob(f'../data/smartfallmm/old/skeleton/*{activity_code}*.csv')

for j, file in enumerate(sorted(files)):
    fig = None
    try:
        df = pd.read_csv(file)
        data = df.values[:, -3:]
        best_corr = -np.inf
        best_window = None
        best_idx = None
        # Slide a window of the same length as watch_gt
        for i in range(0, data.shape[0] - watch_gt.shape[0] + 1):
            window = data[i:i+watch_gt.shape[0], :]
            corr_axes = []
            for axis in range(3):
                corr = np.correlate(window[:, axis] - window[:, axis].mean(),
                                    watch_gt[:, axis] - watch_gt[:, axis].mean(), mode='valid')
                corr_axes.append(corr[0])
            mean_corr = np.mean(corr_axes)
            if mean_corr > best_corr:
                best_corr = mean_corr
                best_window = window
                best_idx = i
        if best_idx is None:
            print(f"No valid window found for {os.path.basename(file)}")
            continue
        
        best_window_df = df.iloc[best_idx:best_idx+window_length+1, :]
        best_window_df.to_csv(f"{out_dir}/{os.path.basename(file)}", index = False)

        fig = plt.figure(figsize=(12, 7))
        plt.title(f"Full data from {os.path.basename(file)} with best window highlighted (index {best_idx})")
        for axis in range(3):
            plt.plot(data[:, axis], label=f'Axis {axis+1}', alpha=0.5)
            plt.plot(np.arange(best_idx, best_idx + watch_gt.shape[0]), 
                    data[best_idx:best_idx + watch_gt.shape[0], axis], 
                    color=f'C{axis}', linewidth=3, label=f'Best Window Axis {axis+1}')
        plt.legend()

        plot_name = f"{os.path.splitext(os.path.basename(file))[0]}.png"
        plt.savefig(os.path.join(visualization_dir, plot_name))
    except Exception as exc:
        print(f"Error processing {os.path.basename(file)}: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)