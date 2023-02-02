from typing import Dict, List, Optional, TypeVar
import torch.nn as nn
import torch
import numpy as np

T = TypeVar("T")


class PatchedLinear(nn.Module):
    """A patched linear module.

    Attributes:
        linear (nn.Linear): The original linear module.
        linear_down (nn.Linear): The low-rank approximation module.
        linear_up (nn.Linear): The low-rank approximation module.
        scale (float): The scale of the low-rank approximation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_weight: torch.Tensor,
        init_bias: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        rank=32,
        scale: float = 1.0,
    ):
        """Create a patched linear module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            init_weight (torch.Tensor): The initial weight of the linear module.
            init_bias (torch.Tensor, optional): The initial bias of the linear module. Defaults to None.
            rank (int, optional): The rank of the low-rank approximation. Defaults to 4.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, init_bias is not None)
        self.linear_down = nn.Linear(in_features, rank, bias=False)
        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
        self.linear_up = nn.Linear(rank, out_features, bias=False)
        self.scale = scale

        nn.init.normal_(self.linear_down.weight, std=1 / rank**2)
        nn.init.zeros_(self.linear_up.weight)

        self.linear.weight = init_weight
        if init_bias is not None:
            self.linear.bias = init_bias  # type: ignore

    def forward(self, input):
        return (
            self.linear(input)
            + self.linear_up(self.dropout(self.activation(self.linear_down(input))))
            * self.scale
        )


class PatchedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        init_weight: torch.Tensor,
        init_bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.1,
        rank: int = 32,
        scale: float = 1.0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        self.conv.weight = init_weight
        if init_bias is not None:
            self.conv.bias = init_bias

        self.conv_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.conv_up = nn.Conv2d(
            in_channels=rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.scale = scale

        nn.init.normal_(self.conv_down.weight, std=1 / rank)
        nn.init.zeros_(self.conv_up.weight)

    def forward(self, input):
        return (
            self.conv(input)
            + self.conv_up(self.activation(self.dropout(self.conv_down(input))))
            * self.scale
        )


def create_trainable_patched_linear(
    linear_module: nn.Linear, rank=32, scale: float = 1
) -> PatchedLinear:
    """Create a patched linear module from a linear module.

    Args:
        linear_module (nn.Linear): The linear module to be patched.
        rank (int, optional): The rank of the low-rank approximation. Defaults to 4.

    Returns:
        PatchedLinear: The patched linear module.
    """

    patched_linear = PatchedLinear(
        linear_module.in_features,
        linear_module.out_features,
        linear_module.weight,
        linear_module.bias,
        rank=rank,
        scale=scale,
    )

    patched_linear.to("cuda", dtype=torch.float16)  # type: ignore

    patched_linear.linear_up.weight.requires_grad = True
    patched_linear.linear_down.weight.requires_grad = True

    return patched_linear


def create_inference_patched_linear(
    linear_module: nn.Linear, rank=32, scale: float = 1
) -> PatchedLinear:
    """Create a patched linear module from a linear module.

    Args:
        linear_module (nn.Linear): The linear module to be patched.
        rank (int, optional): The rank of the low-rank approximation. Defaults to 4.

    Returns:
        PatchedLinear: The patched linear module.
    """

    patched_linear = PatchedLinear(
        linear_module.in_features,
        linear_module.out_features,
        linear_module.weight,
        linear_module.bias,
        rank=rank,
        scale=scale,
        dropout=0,
    )

    patched_linear.to("cuda", dtype=torch.float16)  # type: ignore

    return patched_linear


def create_trainable_patched_conv2d(
    conv_module: nn.Conv2d, rank=32, scale: float = 1
) -> PatchedConv2d:
    """Create a patched conv2d module from a conv2d module.

    Args:
        conv_module (nn.Conv2d): The conv2d module to be patched.
        rank (int, optional): The rank of the low-rank approximation. Defaults to 4.

    Returns:
        PatchedConv2d: The patched conv2d module.
    """

    patched_conv2d = PatchedConv2d(
        conv_module.in_channels,
        conv_module.out_channels,
        conv_module.kernel_size,
        conv_module.weight,
        conv_module.bias,
        conv_module.stride,  # type: ignore
        conv_module.padding,  # type: ignore
        conv_module.dilation,  # type: ignore
        conv_module.groups,
        rank=rank,
        scale=scale,
    )

    patched_conv2d.to("cuda", dtype=torch.float16)  # type: ignore

    patched_conv2d.conv_up.weight.requires_grad = True
    patched_conv2d.conv_down.weight.requires_grad = True

    return patched_conv2d


def create_inference_patched_conv2d(
    conv_module: nn.Conv2d, rank=32, scale: float = 1
) -> PatchedConv2d:
    """Create a patched conv2d module from a conv2d module.

    Args:
        conv_module (nn.Conv2d): The conv2d module to be patched.
        rank (int, optional): The rank of the low-rank approximation. Defaults to 4.

    Returns:
        PatchedConv2d: The patched conv2d module.
    """

    patched_conv2d = PatchedConv2d(
        conv_module.in_channels,
        conv_module.out_channels,
        conv_module.kernel_size,
        conv_module.weight,
        conv_module.bias,
        conv_module.stride,  # type: ignore
        conv_module.padding,  # type: ignore
        conv_module.dilation,  # type: ignore
        conv_module.groups,
        rank=rank,
        scale=scale,
        dropout=0,
    )

    patched_conv2d.to("cuda", dtype=torch.float16)  # type: ignore

    return patched_conv2d


def create_linear_from_patched_linear(patched_linear: PatchedLinear) -> nn.Linear:
    """Create a linear module from a patched linear module.

    Args:
        patched_linear (PatchedLinear): The patched linear module.

    Returns:
        nn.Linear: The linear module.
    """

    linear = nn.Linear(
        patched_linear.linear.in_features,
        patched_linear.linear.out_features,
        patched_linear.linear.bias is not None,
    )

    linear.weight = patched_linear.linear.weight
    linear.bias = patched_linear.linear.bias

    return linear


def set_submodule_from_name(
    module: nn.Module, module_to_inject: nn.Module, submodule_name: str
) -> None:
    """Change a module by name.

    Args:
        module (nn.Module): The module to be changed.
        path_to_submodule (str): The path to the submodule.
        module_to_inject (nn.Module): The module to be injected.
    """
    path_to_submodule = submodule_name.split(".")

    if len(path_to_submodule) == 1:
        setattr(module, path_to_submodule[0], module_to_inject)
    else:
        set_submodule_from_name(
            getattr(module, path_to_submodule[0]),
            module_to_inject,
            ".".join(path_to_submodule[1:]),
        )


def get_submodule_from_name(module: nn.Module, submodule_name: str) -> nn.Module:
    """Get a submodule from a module by name.

    Args:
        module (nn.Module): The module to be searched.
        name (str): The name of the submodule.

    Returns:
        nn.Module: The submodule.
    """
    path_to_submodule = submodule_name.split(".")
    if len(path_to_submodule) == 1:
        return getattr(module, path_to_submodule[0])
    else:
        return get_submodule_from_name(
            getattr(module, path_to_submodule[0]), ".".join(path_to_submodule[1:])
        )


def inject_patched_linear_into_submodule(
    module: nn.Module, submodule_name: str, rank: int = 32, scale: float = 1, trainable: bool = True
) -> None:
    """Inject a patched linear module into a submodule."""

    submodule = get_submodule_from_name(module, submodule_name)
    if isinstance(submodule, nn.Linear):
        set_submodule_from_name(
            module,
            create_trainable_patched_linear(submodule, rank=rank, scale=scale) if trainable else create_inference_patched_linear(submodule, rank=rank, scale=scale),
            submodule_name,
        )
    else:
        raise ValueError(f"Submodule {submodule_name} is not a linear module.")


def inject_patched_conv2d_into_submodule(
    module: nn.Module, submodule_name: str, rank: int = 32, scale: float = 1, trainable: bool = True
) -> None:
    """Inject a patched linear module into a submodule."""

    submodule = get_submodule_from_name(module, submodule_name)
    if isinstance(submodule, nn.Conv2d):
        set_submodule_from_name(
            module,
            create_trainable_patched_conv2d(submodule, rank=rank, scale=scale) if trainable else create_inference_patched_conv2d(submodule, rank=rank, scale=scale),
            submodule_name,
        )
    else:
        raise ValueError(f"Submodule {submodule_name} is not a linear module.")


def remove_patched_linear_from_submodule(
    module: nn.Module, submodule_name: str
) -> None:
    """Remove a patched linear module from a submodule."""

    submodule = get_submodule_from_name(module, submodule_name)
    if submodule.__class__.__name__ == "PatchedLinear":
        set_submodule_from_name(
            module, create_linear_from_patched_linear(submodule), submodule_name  # type: ignore
        )
    else:
        raise ValueError(f"Submodule {submodule_name} is not a patched linear module.")


def remove_patched_conv2d_from_submodule(
    module: nn.Module, submodule_name: str
) -> None:
    """Remove a patched linear module from a submodule."""

    submodule = get_submodule_from_name(module, submodule_name)
    if submodule.__class__.__name__ == "PatchedConv2d":
        set_submodule_from_name(
            module, create_conv2d_from_patched_conv2d(submodule), submodule_name  # type: ignore
        )
    else:
        raise ValueError(f"Submodule {submodule_name} is not a patched linear module.")


def inject_patched_linear_all_layers(
    model: nn.Module, rank: int = 32, scale: float = 1, trainable: bool = True
) -> None:
    """Inject patched linear modules into the model.

    Args:
        model (nn.Module): The model to be patched.
        rank (int, optional): The rank of the low-rank approximation. Defaults to 4.

    """

    for submodule_name, submodule in model.named_modules():
        if isinstance(submodule, nn.Linear):
            inject_patched_linear_into_submodule(
                model, submodule_name, rank=rank, scale=scale, trainable=trainable
            )


def inject_patched_conv2d_all_layers(
    model: nn.Module, rank: int = 32, scale: float = 1, trainable: bool = True
) -> None:
    """Inject patched linear modules into the model.

    Args:
        model (nn.Module): The model to be patched.
        rank (int, optional): The rank of the low-rank approximation. Defaults to 4.

    """

    for submodule_name, submodule in model.named_modules():
        if isinstance(submodule, nn.Conv2d):
            inject_patched_conv2d_into_submodule(
                model, submodule_name, rank=rank, scale=scale, trainable=trainable
            )


def remove_patched_linear_all_layers(model: nn.Module) -> None:
    """Remove patched linear modules from the model.

    Args:
        model (nn.Module): The model to be patched.
    """

    for submodule_name, submodule in model.named_modules():
        if submodule.__class__.__name__ == "PatchedLinear":
            remove_patched_linear_from_submodule(model, submodule_name)


def remove_patched_conv2d_all_layers(model: nn.Module) -> None:
    """Remove patched linear modules from the model.

    Args:
        model (nn.Module): The model to be patched.
    """

    for submodule_name, submodule in model.named_modules():
        if submodule.__class__.__name__ == "PatchedConv2d":
            remove_patched_conv2d_from_submodule(model, submodule_name)


def save_patch_weights(model: nn.Module, path: str) -> None:
    """Save the weights of the patched linear modules.

    Args:
        model (nn.Module): The model to be patched.
        path (str): The path to the file where the weights are saved.
    """

    patch_weights = {}
    for submodule_name, submodule in model.named_modules():
        if submodule.__class__.__name__ == "PatchedLinear":
            patch_weights[submodule_name] = {
                "linear_up": submodule.linear_up.weight.detach().cpu().numpy(),  # type: ignore
                "linear_down": submodule.linear_down.weight.detach().cpu().numpy(),  # type: ignore
            }
        elif submodule.__class__.__name__ == "PatchedConv2d":
            patch_weights[submodule_name] = {
                "conv_up": submodule.conv_up.weight.detach().cpu().numpy(),  # type: ignore
                "conv_down": submodule.conv_down.weight.detach().cpu().numpy(),  # type: ignore
            }

    torch.save(patch_weights, path)


def set_patch_weight(model: nn.Module, patch_weight: Dict[str, np.ndarray]) -> None:
    """Set the weights of the patched linear modules.

    Args:
        model (nn.Module): The model to be patched.
        patch_weight (Dict[str, np.ndarray]): The weights to be set.
    """

    for submodule_name, weights in patch_weight.items():
        submodule = get_submodule_from_name(model, submodule_name)
        if submodule.__class__.__name__ == "PatchedLinear":
            try:
                submodule.linear_up.weight.data = torch.from_numpy(  # type: ignore
                    weights["linear_up"]
                ).to(
                    submodule.linear_up.weight.device  # type: ignore
                )
                submodule.linear_down.weight.data = torch.from_numpy(  # type: ignore
                    weights["linear_down"]
                ).to(
                    submodule.linear_down.weight.device  # type: ignore
                )
            except KeyError:
                submodule.linear_up.weight.data = torch.from_numpy(  # type: ignore
                    weights["lora_up"]
                ).to(
                    submodule.linear_up.weight.device  # type: ignore
                )
                submodule.linear_down.weight.data = torch.from_numpy(  # type: ignore
                    weights["lora_down"]
                ).to(
                    submodule.linear_down.weight.device  # type: ignore
                )

        elif submodule.__class__.__name__ == "PatchedConv2d":
            submodule.conv_up.weight.data = torch.from_numpy(  # type: ignore
                weights["conv_up"]
            ).to(
                submodule.conv_up.weight.device  # type: ignore
            )
            submodule.conv_down.weight.data = torch.from_numpy(  # type: ignore
                weights["conv_down"]
            ).to(
                submodule.conv_down.weight.device  # type: ignore
            )
        else:
            raise ValueError(f"Submodule {submodule_name} is not a patched module. Did you forget to inject it?")




def load_patch_weights(model: nn.Module, path: str) -> None:
    """Load the weights of the patched linear modules.

    Args:
        model (nn.Module): The model to be patched.
        path (str): The path to the file where the weights are saved.
    """

    patch_weight = torch.load(path)
    set_patch_weight(model, patch_weight)


def extract_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Extract the trainable parameters from a model.

    Args:
        model (nn.Module): The model to be patched.

    Returns:
        List[nn.Parameter]: The trainable parameters.
    """

    return list(filter(lambda x: x.requires_grad, model.parameters()))


def set_scale(model: nn.Module, scale: float) -> None:
    """Set the scale of the patched linear modules.

    Args:
        model (nn.Module): The model to be patched.
        scale (float): The scale to be set.
    """

    for _, submodule in model.named_modules():
        if submodule.__class__.__name__ == "PatchedLinear":
            submodule.scale = scale  # type: ignore
        elif submodule.__class__.__name__ == "PatchedConv2d":
            submodule.scale = scale  # type: ignore