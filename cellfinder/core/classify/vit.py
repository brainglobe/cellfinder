from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

from keras import (
    KerasTensor as Tensor,
)
from keras import Model
from keras import layers
from keras import optimizers
from keras import ops as K


class VITConfig:
    num_layers: int
    hidden_dim: int
    num_heads: int
    expanding_factor: int
    patch_size: Tuple[int, int, int]
    layer_norm_eps: float = 1e-6


network_type = Literal[
    "vit-4-layer",
    "vit-8-layer",
    "vit-12-layer",
    "vit-24-layer",
    "vit-32-layer",
]


vit_configs: Dict[network_type, VITConfig] = {
    "vit-4-layer": VITConfig(
        num_layers=4,
        hidden_dim=64,
        num_heads=8,
        expanding_factor=4,
        patch_size=(8, 8, 4),
    ),
    "vit-8-layer": VITConfig(
        num_layers=8,
        hidden_dim=256,
        num_heads=8,
        expanding_factor=4,
        patch_size=(8, 8, 4),
    ),
    "vit-12-layer": VITConfig(
        num_layers=12,
        hidden_dim=768,
        num_heads=8,
        expanding_factor=4,
        patch_size=(8, 8, 4),
    ),
    "vit-24-layer": VITConfig(
        num_layers=24,
        hidden_dim=1024,
        num_heads=8,
        expanding_factor=4,
        patch_size=(8, 8, 4),
    ),
    "vit-32-layer": VITConfig(
        num_layers=32,
        hidden_dim=4096,
        num_heads=8,
        expanding_factor=4,
        patch_size=(8, 8, 4),
    ),
}


class PositionalEmbeddings(layers.Layer):
    """
    Add positional embeddings to the input tensor.
    This seems to be not implemented in keras yet, so we have to do it

    :param int embedding_dim: The dimension of the embeddings
    """
    def __init__(
        self,
        embedding_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(
        self,
        input_shape: Tuple[int],
    ):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embedding_dim
        )
        self.positions = K.arange(0, num_tokens, 1)

    def call(
        self,
        inputs,
    ):
        return K.broadcast_to(
            self.position_embedding(self.positions),
            K.shape(inputs)
        )


def attention_block(
    inputs,
    layer_norm_eps: float,
    num_heads: int,
    hidden_dim: int,
    name="attention_block",
):
    """
    Apply a multi-head attention block.

    :param inputs: The input tensor
    :param layer_norm_eps: The epsilon value for the layer normalization
    :param num_heads: The number of heads in the multi-head attention
    :param hidden_dim: The hidden dimension of the multi-head attention
    :param name: The name of the block
    :return: The residual-output of the multi-head attention block
    """
    normalized_inputs = layers.LayerNormalization(
        epsilon=layer_norm_eps,
        name=f"{name}--layer_norm",
    )(inputs)
    return layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_dim // num_heads,
        name=f"{name}--mha"
    )(normalized_inputs, normalized_inputs) 


def mlp_block(
    inputs,
    hidden_dim: int,
    layer_norm_eps: float,
    expanding_factor: int,
    name: str = "mlp_block",
):
    """
    Apply a multi-layer perceptron block.

    :param inputs: The input tensor
    :param hidden_dim: The hidden dimension of the MLP
    :param layer_norm_eps: The epsilon value for the layer normalization
    :param expanding_factor: The factor by which the hidden dimension is
    expanded in the MLP
    :param name: The name of the block
    :return: The residual-output of the MLP block
    """
    normalized_inputs = layers.LayerNormalization(
        epsilon=layer_norm_eps,
        name=f"{name}--layer_norm",
    )(inputs)
    hidden_states = layers.Dense(
        units=hidden_dim * expanding_factor,
        activation=K.gelu,
        name=f"{name}--up",
    )(normalized_inputs)
    return layers.Dense(
        units=hidden_dim,
        name=f"{name}--down",
    )(hidden_states)


def transformer_block(
    residual_stream,
    layer_norm_eps: float = 1e-6, 
    num_heads: int = 8,
    hidden_dim: int = 128,
    expanding_factor: int = 4,
    name: str = "transformer_block",
):
    """
    Apply a transformer block a.k.a. transformer layer.

    :param residual_stream: The input tensor
    :param layer_norm_eps: The epsilon value for the layer normalization
    :param num_heads: The number of heads in the multi-head attention
    :param hidden_dim: The hidden dimension of the multi-head attention
    :param expanding_factor: The factor by which the hidden dimension is
    expanded in the MLP
    :param name: The name of the block
    :return: The residual-output of the transformer block
    """

    attention_outputs = attention_block(
        residual_stream,
        layer_norm_eps=layer_norm_eps,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        name=f"{name}--attention_block"
    )

    residual_stream = layers.Add()([
        residual_stream,
        attention_outputs
    ])

    mlp_outputs = mlp_block(
        residual_stream,
        hidden_dim=hidden_dim,
        layer_norm_eps=layer_norm_eps,
        expanding_factor=expanding_factor,
        name=f"{name}--mlp_block",
    )

    # Skip connection
    residual_stream = layers.Add()([
        residual_stream,
        mlp_outputs
    ])

    return residual_stream


def build_model(
    input_shape: Tuple[int, int, int, int] = (50, 50, 20, 2),
    network_depth: network_type = "24-layer",
    optimizer: Optional[optimizers.Optimizer] = None,
    learning_rate: float = 0.0005,
    loss: str = "categorical_crossentropy",
    metrics: List[str] = ["accuracy"],
    num_classes: int = 2,
    embedding_dim: int = 128,
    classification_activation: str = "softmax",
) -> Model:
    """
    Build a Vision Transformer model.

    Mostly follows the signature of the ResNet model, but with additional
    parameters for the Vision Transformer.
    """
    config = vit_configs[network_depth]
    
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = layers.Conv3D(
        name="patch_embedding",
        filters=embedding_dim,
        kernel_size=config.patch_size,
        strides=config.patch_size,
        padding="VALID",
    )(inputs)
    patches = layers.Reshape(
        target_shape=(-1, embedding_dim)
    )(patches)

    # Add positional embeddings
    positional_embeddings = PositionalEmbeddings(
        embedding_dim=embedding_dim
    )(patches)

    residual_stream = layers.Add()([
        patches,
        positional_embeddings,
    ])

    # Create multiple layers of the Transformer block.
    for layer_idx in range(config.num_layers):
        residual_stream = transformer_block(
            residual_stream,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            expanding_factor=config.expanding_factor,
            layer_norm_eps=config.layer_norm_eps,
            name=f"transformer_block_{layer_idx}",
        )

    normalized_stream = layers.LayerNormalization(
        epsilon=config.layer_norm_eps,
        name="pre_logits_norm"
    )(residual_stream)
    flat_feature_vector = layers.GlobalAvgPool1D()(normalized_stream)

    outputs = layers.Dense(
        units=num_classes,
        activation=classification_activation,
    )(flat_feature_vector)

    model = Model(
        inputs=inputs,
        outputs=outputs,
    )
    
    if optimizer is None:
        optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model
