import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import glob
import os

# Original 32-joint labels
ORIGINAL_JOINT_LABELS = [
    "pelvis", "spine1", "spine2", "neck",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand", "left_handtip", "left_thumb",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand", "right_handtip", "right_thumb",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "head", "nose", "eye_left", "ear_left", "eye_right", "ear_right"
]

# 24 SMPL joint names
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]

# Map SMPL joint names to original 32-joint indices (spine3 -> neck as fallback)
_orig_label_to_idx = {label: i for i, label in enumerate(ORIGINAL_JOINT_LABELS)}
SMPL_TO_ORIGINAL_IDX = [
    _orig_label_to_idx.get(name, _orig_label_to_idx["neck"])
    for name in SMPL_JOINT_NAMES
]

# Skeleton bone connections in SMPL joint indices
SMPL_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]

NUM_JOINTS = 32
SKELETON_COLS = NUM_JOINTS * 3   # 96


def load_skeleton(file_path: str) -> np.ndarray:
    """Load skeleton CSV and return [T, 96] float array (last 96 columns)."""
    df = pd.read_csv(file_path, index_col=False, header=0)
    return df.values[:, -SKELETON_COLS:].astype(np.float32)


def mean_joint_trajectory(skeleton: np.ndarray) -> np.ndarray:
    """
    Return the trajectory of joint 8 (spine3 / right_ankle area) across frames.
    Args:
        skeleton: [T, 96]
    Returns:
        traj: [T, 3]
    """
    return skeleton.reshape(-1, NUM_JOINTS, 3)[:, 8, :]


def correlate_windows(data_traj: np.ndarray, gt_traj: np.ndarray):
    """
    Slide a window over data_traj and find the best match with gt_traj
    using mean cross-correlation across all 3 axes.

    Args:
        data_traj : [T, 3]  — full recording centroid
        gt_traj   : [W, 3]  — ground-truth window centroid
    Returns:
        best_idx  : int
        best_corr : float
    """
    W = gt_traj.shape[0]
    best_corr = -np.inf
    best_idx = None
    for i in range(0, data_traj.shape[0] - W + 1):
        window = data_traj[i:i + W]
        corr_axes = []
        for axis in range(3):
            c = np.correlate(
                window[:, axis] - window[:, axis].mean(),
                gt_traj[:, axis] - gt_traj[:, axis].mean(),
                mode='valid'
            )
            corr_axes.append(c[0])
        mean_corr = np.mean(corr_axes)
        if mean_corr > best_corr:
            best_corr = mean_corr
            best_idx = i
    return best_idx, best_corr


def plot_skeleton_3d(ax, frame: np.ndarray, title: str = ""):
    """
    Draw 24 SMPL joints on a 3-D axes.
    Args:
        ax    : matplotlib Axes3D
        frame : [96] — one frame of the full 32-joint skeleton
        title : axes title
    """
    joints_all = frame.reshape(NUM_JOINTS, 3)
    joints_24  = joints_all[SMPL_TO_ORIGINAL_IDX]   # [24, 3]

    # data Y (col 1) is height, negated to flip right-side up → plot Z; data X→plot X, data Z→plot Y
    ax.scatter(joints_24[:, 0], joints_24[:, 2], -joints_24[:, 1],
               c='steelblue', s=30, zorder=5)

    for i, name in enumerate(SMPL_JOINT_NAMES):
        ax.text(joints_24[i, 0], joints_24[i, 2], -joints_24[i, 1],
                name, fontsize=5, color='dimgray')

    for (a, b) in SMPL_CONNECTIONS:
        ax.plot([joints_24[a, 0], joints_24[b, 0]],
                [joints_24[a, 2], joints_24[b, 2]],
                [-joints_24[a, 1], -joints_24[b, 1]],
                'b-', linewidth=1.2)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=8)


# ---------------------------------------------------------------------------
activity_code = 'A08'

# Ground truth: use joint 8
gt_skeleton_file_name = f'../data/smartfallmm_original/old/skeleton/S05{activity_code}T03.csv'
gt_skeleton = load_skeleton(gt_skeleton_file_name)
print(gt_skeleton.shape)
gt_skeleton = gt_skeleton[45:280, :]                 # [130, 96]
gt_traj = mean_joint_trajectory(gt_skeleton)     # [130, 3]
window_length = gt_traj.shape[0]

print(f"GT shape: {gt_skeleton.shape}  |  centroid traj: {gt_traj.shape}")

out_dir = "../data/smartfallmm_test/old/skeleton"
os.makedirs(out_dir, exist_ok=True)
visualization_dir = f"../visualizations/old/skeleton/{activity_code}"
os.makedirs(visualization_dir, exist_ok=True)

files = glob.glob(f'../data/smartfallmm_original/old/skeleton/*{activity_code}*.csv')

for j, file in enumerate(sorted(files)):
    fig = None
    try:
        df = pd.read_csv(file)
        skeleton = df.values[:, -SKELETON_COLS:].astype(np.float32)  # [T, 96]
        data_traj = mean_joint_trajectory(skeleton)                   # [T, 3]

        best_idx, best_corr = correlate_windows(data_traj, gt_traj)

        if best_idx is None:
            print(f"No valid window found for {os.path.basename(file)} , timestamps:{skeleton.shape[0]}")
            continue

        # Save aligned window
        best_window_df = df.iloc[best_idx:best_idx + window_length + 1, :]
        best_window_df.to_csv(f"{out_dir}/{os.path.basename(file)}", index=False)

        # ---- Animation: centroid trajectory + 3-D skeleton ----
        if os.path.basename(gt_skeleton_file_name) == os.path.basename(file):
            fig = plt.figure(figsize=(14, 7))
            fig.suptitle(f"{os.path.basename(file)}  |  best window @ {best_idx}  |  corr={best_corr:.3f}")

            ax_traj = fig.add_subplot(1, 2, 1)
            labels = ['X', 'Y', 'Z']
            colors = ['C0', 'C1', 'C2']
            for axis in range(3):
                ax_traj.plot(data_traj[:, axis], alpha=0.4, color=colors[axis], label=f'{labels[axis]} full')
                ax_traj.plot(
                    np.arange(best_idx, best_idx + window_length),
                    data_traj[best_idx:best_idx + window_length, axis],
                    color=colors[axis], linewidth=2.5, label=f'{labels[axis]} window'
                )
            ax_traj.set_title("Joint 8 trajectory")
            ax_traj.set_xlabel("Frame"); ax_traj.set_ylabel("Position")
            ax_traj.legend(fontsize=7)
            vline = ax_traj.axvline(x=best_idx, color='k', linestyle='--', linewidth=1)

            ax_skel = fig.add_subplot(1, 2, 2, projection='3d')

            window_frames = skeleton[best_idx:best_idx + window_length]  # [W, 96]
            # Compute fixed axis limits from the whole window
            joints_window = window_frames.reshape(-1, NUM_JOINTS, 3)[:, SMPL_TO_ORIGINAL_IDX, :]
            pad = 0.1
            xlim = (joints_window[:, :, 0].min() - pad, joints_window[:, :, 0].max() + pad)  # data X
            ylim = (joints_window[:, :, 2].min() - pad, joints_window[:, :, 2].max() + pad)  # data Z
            zlim = (-joints_window[:, :, 1].max() - pad, -joints_window[:, :, 1].min() + pad)  # data Y negated (up)

            def update(frame_offset):
                ax_skel.cla()
                abs_frame = best_idx + frame_offset
                plot_skeleton_3d(ax_skel, window_frames[frame_offset],
                                title=f"Skeleton @ frame {abs_frame} (24 SMPL joints)")
                ax_skel.set_xlim(xlim); ax_skel.set_ylim(ylim); ax_skel.set_zlim(zlim)
                ax_skel.view_init(elev=20, azim=30)
                vline.set_xdata([abs_frame, abs_frame])
                return []

            ani = animation.FuncAnimation(
                fig, update, frames=window_length, interval=50, blit=False
            )

            plt.tight_layout()
            base_name = os.path.splitext(os.path.basename(file))[0]
            mp4_path = os.path.join(visualization_dir, f"{base_name}.mp4")
            gif_path = os.path.join(visualization_dir, f"{base_name}.gif")
            try:
                ani.save(mp4_path, writer=animation.FFMpegWriter(fps=30, bitrate=800))
            except Exception:
                ani.save(gif_path, writer=animation.PillowWriter(fps=30))

    except Exception as exc:
        print(f"Error processing {os.path.basename(file)}: {exc}")
    # finally:
    #     if fig is not None:
    #         plt.close(fig)
