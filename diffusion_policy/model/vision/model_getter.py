import torch
import torchvision


def patch_resnet_input_channels(resnet, in_channels):
    """
    Modify ResNet conv1 to support 1 / 3 / 4 input channels.
    For 3 channels, keep original conv1.
    """
    if in_channels == 3:
        return resnet

    old_conv = resnet.conv1

    new_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    with torch.no_grad():
        if in_channels == 1:
            # RGB weights average -> Depth weight
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        elif in_channels == 4:
            # First 3 channels use RGB pretrained weights
            new_conv.weight[:, :3, :, :] = old_conv.weight

            # 4th channel depth initialized as RGB average
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        else:
            raise ValueError(f"Unsupported in_channels: {in_channels}")

        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    resnet.conv1 = new_conv
    return resnet


def get_resnet(name, weights=None, in_channels=3, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    in_channels: 1, 3, or 4
    """
    if (weights == "r3m") or (weights == "R3M"):
        resnet = get_r3m(name=name, **kwargs)
    else:
        func = getattr(torchvision.models, name)
        resnet = func(weights=weights, **kwargs)
        resnet.fc = torch.nn.Identity()

    resnet = patch_resnet_input_channels(
        resnet=resnet,
        in_channels=in_channels
    )

    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model