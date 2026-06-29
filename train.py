import argparse
import tensorflow as tf

from data.dataset import prepare_dataset
from model.network import build_model
from losses import (
    denoising_loss,
    metric_mse,
    metric_laplacian,
    metric_chord_length,
    metric_start_point,
    metric_end_point,
)
from visualize import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the inception-based 2D curve denoising autoencoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--noisy_dataset", required=True,
        help="Path to the directory containing noisy training curves."
    )
    parser.add_argument(
        "--clean_dataset", required=True,
        help="Path to the directory containing clean training curves."
    )
    parser.add_argument(
        "--epochs", type=int, default=4000,
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size."
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience (epochs without improvement)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001,
        help="Adam optimizer learning rate."
    )
    parser.add_argument(
        "--save_path", type=str, default="model/denoising_autoencoder.h5",
        help="Path to save the trained model."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load data ---
    print("Loading dataset...")
    X, y = prepare_dataset(args.noisy_dataset, args.clean_dataset)
    print(f"Loaded {len(X)} curve pairs. Input shape: {X.shape}")

    # --- Build model ---
    model = build_model()
    model.summary()

    # --- Compile ---
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        weight_decay=1e-6,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=denoising_loss,
        metrics=[
            metric_mse,
            metric_laplacian,
            metric_chord_length,
            metric_start_point,
            metric_end_point,
        ]
    )

    # --- Train ---
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=args.patience,
        restore_best_weights=True,
    )

    print("Starting training...")
    history = model.fit(
        X, y,
        shuffle=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=[early_stopping],
    )

    # --- Save model ---
    model.save(args.save_path)
    print(f"Model saved to: {args.save_path}")

    # --- Plot training history ---
    plot_training_history(history, save_dir=".")


if __name__ == "__main__":
    main()
