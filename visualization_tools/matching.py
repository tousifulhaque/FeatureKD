import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
activity_code = 'A09'
# Load the ground truth window
#watch_gt = pd.read_csv('../data/smartfallmm_original/young/accelerometer/watch/S30A12T03.csv').values[:, -3:][:100]
#watch_gt = pd.read_csv('../data/smartfallmm_original/young/accelerometer/watch/S48A13T01.csv').values[:, -3:][125:176, :]
#watch_gt = pd.read_csv('../data/smartfallmm_original/young/accelerometer/watch/S52A14T04.csv').values[:, -3:][175:250, :]
#watch_gt = pd.read_csv('../data/smartfallmm_original/young/accelerometer/watch/S50A11T03.csv').values[180:225, -3:]
#watch_gt = pd.read_csv('../data/smartfallmm/young/accelerometer/watch/S45A13T02.csv').values[:, -3:][500:560, :]
#watch_gt = pd.read_csv('../data/smartfallmm_original/young/accelerometer/watch/S52A10T01.csv').values[175:225, -3:]
watch_gt = pd.read_csv(f'../data/smartfallmm_original/young/accelerometer/watch/S62{activity_code}T03.csv').values[:, -3:][100:350, :]
#watch_gt = pd.read_csv(f'../data/smartfallmm_original/young/accelerometer/watch/S62A02T03.csv').values[:, -3:][100:, :]
#watch_gt = pd.read_csv(f'../data/smartfallmm_original/young/accelerometer/watch/S62{activity_code}T03.csv').values[:, -3:][100:, :]



window_length = watch_gt.shape[0]
out_dir = "../data/smartfallmm/young/accelerometer/watch"
os.makedirs(out_dir, exist_ok = True)
visualization_dir = f"visualizations_secondattempt/young/{activity_code}"
os.makedirs(visualization_dir, exist_ok=True)
# Find all files with 'A13' in the filename
files = glob.glob(f'../data/smartfallmm_original/young/accelerometer/watch/*{activity_code}*.csv')

for j, file in enumerate(sorted(files)):
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

            
    fig = None
    try:
        best_window_df = df.iloc[best_idx:best_idx+window_length+1, :]
        best_window_df.iloc[10:-10, :].to_csv(f"{out_dir}/{os.path.basename(file)}", index = False)

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
    except:
        pass
    finally:
        if fig is not None:
            plt.close(fig)