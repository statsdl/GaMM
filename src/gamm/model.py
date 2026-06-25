from __future__ import annotations

from typing import Any


def build_gamm_model(
    input_shape: tuple[int, int],
    pred_len: int = 1,
    norm_type: str = "B",
    activation: str = "relu",
    n_block: int = 1,
    dropout: float = 0.0,
    ff_dim: int = 64,
    target_slice: int | slice | None = None,
):
    """Build the GaMMixer model from the reference workflow using Keras."""

    tf, layers = _tensorflow()
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(int(n_block)):
        x = _res_block(x, norm_type=norm_type, activation=activation, dropout=dropout, ff_dim=int(ff_dim))

    if target_slice is not None:
        x = x[:, :, target_slice]

    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Flatten(input_shape=input_shape[1:])(x)
    outputs = layers.Dense(pred_len)(x)
    return tf.keras.Model(inputs, outputs)


def compile_gamm_model(
    input_shape: tuple[int, int],
    learning_rate: float = 0.001,
    **model_kwargs: Any,
):
    """Build and compile a GaMMixer model with Adam and MSE loss."""

    tf, _ = _tensorflow()
    model = build_gamm_model(input_shape=input_shape, **model_kwargs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def _res_block(inputs, norm_type: str, activation: str, dropout: float, ff_dim: int):
    tf, layers = _tensorflow()
    norm = layers.LayerNormalization if norm_type == "L" else layers.BatchNormalization

    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = norm(axis=[-2, -1])(res)
    x = layers.Dense(ff_dim, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    return x + res


def _tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for GaMM model training. Install with `pip install gamm[model]`."
        ) from exc
    return tf, layers
