
import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Used to enhance channel-wise feature representation.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        self.fc1 = nn.Linear(
            channel,
            channel // reduction
        )

        self.fc2 = nn.Linear(
            channel // reduction,
            channel
        )


    def forward(self, x):

        batch, channels, _, _ = x.size()

        se = nn.functional.adaptive_avg_pool2d(
            x,
            1
        ).view(batch, channels)

        se = torch.relu(
            self.fc1(se)
        )

        se = torch.sigmoid(
            self.fc2(se)
        ).view(batch, channels, 1, 1)

        return x * se



def load_resnet18_senet_teacher(
        checkpoint_path,
        num_classes=5,
        device=None
):

    """
    Load ResNet18 + SENet Teacher model.

    Args:
        checkpoint_path:
            Path to trained teacher checkpoint.

        num_classes:
            Number of ECG classes.

        device:
            cuda or cpu

    Returns:
        Loaded teacher model
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )


    # Initialize ResNet18 backbone
    model = models.resnet18(
        weights=None
    )


    # Replace classifier for ECG classification
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )


    # Load trained checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    # Support both formats:
    # 1) {'model': model}
    # 2) state_dict only

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"]

    else:
        model.load_state_dict(checkpoint)



    model = model.to(device)


    # Teacher is used only for inference during KD
    model.eval()


    # Freeze teacher parameters
    for param in model.parameters():
        param.requires_grad = False


    return model



if __name__ == "__main__":


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    teacher = load_resnet18_senet_teacher(
        checkpoint_path=
        "weights/resnet18_senet_teacher.pt",
        num_classes=5,
        device=device
    )


    total_params = sum(
        p.numel()
        for p in teacher.parameters()
    )


    print(
        f"ResNet18+SENet Teacher Parameters: {total_params:,}"
    )

    print("Teacher loaded successfully.")
