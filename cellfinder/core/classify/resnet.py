from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

from keras import (
    KerasTensor as Tensor,
)
from keras import Model
from keras.initializers import Initializer
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv3D,
    Dense,
    GlobalAveragePooling3D,
    Input,
    MaxPooling3D,
    ZeroPadding3D,
)
from keras.optimizers import Adam, Optimizer

#####################################################################
# Define the types of ResNet

layer_type = Literal[
    "18-layer", "34-layer", "50-layer", "101-layer", "152-layer"
]

resnet_unit_blocks: Dict[layer_type, List[int]] = {
    "18-layer": [2, 2, 2, 2],
    "34-layer": [3, 4, 6, 3],
    "50-layer": [3, 4, 6, 3],
    "101-layer": [3, 4, 23, 3],
    "152-layer": [3, 6, 36, 3],
}

network_residual_bottleneck: Dict[layer_type, bool] = {
    "18-layer": False,
    "34-layer": False,
    "50-layer": True,
    "101-layer": True,
    "152-layer": True,
}
#####################################################################


def build_model(
    shape: Tuple[int, int, int, int] = (50, 50, 20, 2),
    network_depth: layer_type = "18-layer",
    optimizer: Optional[Optimizer] = None,
    learning_rate: float = 0.0005,  # higher rates don't always converge
    loss: str = "categorical_crossentropy",
    metrics: List[str] = ["accuracy"],
    number_classes: int = 2,
    axis: int = 3,
    starting_features: int = 64,
    classification_activation: str = "softmax",
) -> Model:
    blocks, bottleneck = get_resnet_blocks_and_bottleneck(network_depth)

    inputs = Input(shape)
    x = non_residual_block(inputs, starting_features, axis=axis)

    features = starting_features
    for resnet_unit_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = residual_block(
                features,
                resnet_unit_id,
                block_id,
                bottleneck=bottleneck,
                axis=axis,
            )(x)

        features *= 2

    x = GlobalAveragePooling3D(name="final_average_pool")(x)
    x = Dense(
        number_classes,
        activation=classification_activation,
        name="fully_connected",
    )(x)

    # Instantiate existing_model.
    model = Model(inputs=inputs, outputs=x)

    if optimizer is None:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer, loss=loss, metrics=metrics)
    return model


def get_resnet_blocks_and_bottleneck(
    network_depth: layer_type,
) -> Tuple[List[int], bool]:
    """
    Parses dicts, and returns how many resnet blocks are in each unit, along
    with whether they are bottlneck blocks or not

    :param network_depth:
    :return:
    """
    blocks = resnet_unit_blocks[network_depth]
    bottleneck = network_residual_bottleneck[network_depth]
    return blocks, bottleneck


def non_residual_block(
    inputs: Tensor,
    starting_features: int,
    conv_kernel: Tuple[int, int, int] = (7, 7, 3),
    strides: Tuple[int, int, int] = (2, 2, 2),
    padding: int = 3,
    max_pool_size: Tuple[int, int, int] = (3, 3, 2),
    activation: str = "relu",
    use_bias: bool = False,
    bn_epsilon: float = 1e-5,
    pooling_padding: str = "valid",
    axis: int = 3,
) -> Tensor:
    """
    Non-residual unit from He et al. (2015). Corresponds to "conv1" and the
    max pool.
    """

    x = ZeroPadding3D(padding=padding, name="conv1_padding")(inputs)
    x = Conv3D(
        starting_features,
        conv_kernel,
        strides=strides,
        use_bias=use_bias,
        name="conv1",
    )(x)
    x = BatchNormalization(axis=axis, epsilon=bn_epsilon, name="conv1_bn")(x)
    x = Activation(activation, name="conv1_activation")(x)

    x = MaxPooling3D(
        max_pool_size,
        strides=strides,
        padding=pooling_padding,
        name="max_pool",
    )(x)

    return x


def residual_block(
    output_features: Tensor,
    resnet_unit_id: int,
    block_id: int,
    conv_kernel: Tuple[int, int, int] = (3, 3, 3),
    bottleneck_conv_kernel: Tuple[int, int, int] = (1, 1, 1),
    bottleneck: int = False,
    activation: str = "relu",
    use_bias: bool = False,
    kernel_initializer: str = "he_normal",
    bn_epsilon: float = 1e-5,
    axis: int = 3,
) -> Callable[[Tensor], Tensor]:
    """
    Residual unit from He et al. (2015)


    :param int output_features: How many output features
    :param int resnet_unit_id: Which resnet unit
    (e.g. 0 is conv2_x, 1 is conv3_x)
    :param int block_id: Which block in the resnet unit (i.e. in the third
    unit of a 152-layer resnet, block_id=36)
    :param conv_kernel:
    :param bottleneck_conv_kernel:
    :param bool bottleneck: If True, use a bottleneck resnet unit.
    Default False.
    :param activation:
    :param use_bias:
    :param kernel_initializer:
    :param bn_epsilon:
    :param axis:
    :return:
    """

    stride = get_stride(resnet_unit_id, block_id)
    resnet_unit_label = resnet_unit_id + 2

    def f(x: Tensor) -> Tensor:
        if bottleneck:
            y = Conv3D(
                output_features,
                bottleneck_conv_kernel,
                strides=stride,
                use_bias=use_bias,
                name=f"resunit{resnet_unit_label}_block{block_id}_conv_a",
                kernel_initializer=kernel_initializer,
            )(x)
        else:
            y = ZeroPadding3D(
                padding=1,
                name=f"resunit{resnet_unit_label}_block{block_id}_pad_a",
            )(x)

            y = Conv3D(
                output_features,
                conv_kernel,
                strides=stride,
                use_bias=use_bias,
                name=f"resunit{resnet_unit_label}_block{block_id}_conv_a",
                kernel_initializer=kernel_initializer,
            )(y)

        y = BatchNormalization(
            axis=axis,
            epsilon=bn_epsilon,
            name=f"resunit{resnet_unit_label}_block{block_id}_bn_a",
        )(y)

        y = Activation(
            activation,
            name=f"resunit{resnet_unit_label}_block{block_id}_activation_a",
        )(y)

        y = ZeroPadding3D(
            padding=1, name=f"resunit{resnet_unit_label}_block{block_id}_pad_b"
        )(y)

        y = Conv3D(
            output_features,
            conv_kernel,
            use_bias=use_bias,
            name=f"resunit{resnet_unit_label}_block{block_id}_conv_b",
            kernel_initializer=kernel_initializer,
        )(y)

        y = BatchNormalization(
            axis=axis,
            epsilon=bn_epsilon,
            name=f"resunit{resnet_unit_label}_block{block_id}_bn_b",
        )(y)

        if bottleneck:
            y = Activation(
                activation,
                name=f"resunit{resnet_unit_label}_block"
                f"{block_id}_activation_b",
            )(y)

            y = Conv3D(
                output_features * 4,
                bottleneck_conv_kernel,
                use_bias=use_bias,
                name=f"resunit{resnet_unit_label}_block{block_id}_conv_c",
                kernel_initializer=kernel_initializer,
            )(y)

            y = BatchNormalization(
                axis=axis,
                epsilon=bn_epsilon,
                name=f"resunit{resnet_unit_label}_block{block_id}_bn_c",
            )(y)

            identity_shortcut = get_shortcut(
                x,
                resnet_unit_label,
                block_id,
                output_features * 4,
                stride,
                axis=axis,
            )
        else:
            identity_shortcut = get_shortcut(
                x,
                resnet_unit_label,
                block_id,
                output_features,
                stride,
                axis=axis,
            )

        y = Add(name=f"resunit{resnet_unit_label}_block{block_id}_add")(
            [y, identity_shortcut]
        )

        y = Activation(
            activation,
            name=f"resunit{resnet_unit_label}_block{block_id}_activation_c",
        )(y)

        return y

    return f


def get_shortcut(
    inputs: Tensor,
    resnet_unit_label: int,
    block_id: int,
    features: int,
    stride: int,
    use_bias: bool = False,
    kernel_initializer: Union[str, Initializer] = "he_normal",
    bn_epsilon: float = 1e-5,
    axis: int = 3,
) -> Tensor:
    """
    Create shortcut. For none-bottleneck residual units, this is just the
    identity. Otherwise, the input is reshaped to match the output of the
    bottleneck unit

    :param inputs: Input to the residual unit
    :param int resnet_unit_label: Which resnet unit (i.e. 2 is conv2_x)
    :param int block_id: Which block in the resnet unit (i.e. in the third
    unit of a 152-layer resnet, block_id=36)
    :param int features: How many output features
    :param stride: Convolution stride
    :param use_bias:
    :param kernel_initializer:
    :param bn_epsilon:
    :param axis:
    :return: Shortcut tensor, to add to the output of the residual unit
    """

    if block_id == 0:
        shortcut = Conv3D(
            features,
            (1, 1, 1),
            strides=stride,
            use_bias=use_bias,
            name=f"resunit{resnet_unit_label}_block{block_id}_shortcut_conv",
            kernel_initializer=kernel_initializer,
        )(inputs)

        shortcut = BatchNormalization(
            axis=axis,
            epsilon=bn_epsilon,
            name=f"resunit{resnet_unit_label}_block{block_id}_shortcut_bn",
        )(shortcut)
        return shortcut
    else:
        return inputs


def get_stride(resnet_unit_id: int, block_id: int) -> int:
    """
    Determines the convolution stride.

    :param int resnet_unit_id: Which resnet unit
    (e.g. 0 is conv2_x, 1 is conv3_x)
    :param int block_id: Which block in the resnet unit (i.e. in the third
    unit of a 152-layer resnet, block_id=36)
    :return: Stride
    """
    if resnet_unit_id == 0 or block_id != 0:
        return 1
    else:
        return 2
