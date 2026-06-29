import matplotlib.pyplot as plt


def plot_training_history(history, save_dir="."):

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    components = [
        ("loss",                  "Total Loss",         "red"),
        ("metric_mse",            "MSE Loss",           "blue"),
        ("metric_laplacian",      "Laplacian",          "green"),
        ("metric_chord_length",   "Chord Length",       "black"),
        ("metric_start_point",    "Start Point Loss",   "orange"),
        ("metric_end_point",      "End Point Loss",     "pink"),
    ]

    for ax, (key, label, color) in zip(axes, components):
        if key in history.history:
            ax.plot(history.history[key], label=label, color=color)
            ax.set_title(label)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    save_path = f"{save_dir}/training_history.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved training history plot to: {save_path}")


def plot_curve_comparison(noisy, predicted, clean=None, title="Curve denoising result", save_path=None):

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(noisy[:, 0],     noisy[:, 1],     'b--', linewidth=1.0, alpha=0.6, label="Noisy input")
    ax.plot(predicted[:, 0], predicted[:, 1], 'r-',  linewidth=1.5,            label="Predicted (denoised)")

    if clean is not None:
        ax.plot(clean[:, 0], clean[:, 1], 'g-', linewidth=1.5, label="Clean target")

    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved curve comparison to: {save_path}")

    plt.show()
