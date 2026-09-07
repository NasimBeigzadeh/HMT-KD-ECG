
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Depthwise Separable Convolution
# =========================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution used for lightweight CNN design.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )


    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

        return x



# =========================================================
# Channel Attention (CBAM simplified)
# =========================================================

class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        reduced_channels = max(channels // reduction, 1)


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc1 = nn.Conv2d(
            channels,
            reduced_channels,
            kernel_size=1,
            bias=False
        )

        self.fc2 = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
            bias=False
        )


        self.sigmoid = nn.Sigmoid()



    def forward(self,x):

        avg_out = self.fc2(
            F.relu(
                self.fc1(
                    self.avg_pool(x)
                )
            )
        )


        max_out = self.fc2(
            F.relu(
                self.fc1(
                    self.max_pool(x)
                )
            )
        )


        attention = self.sigmoid(avg_out + max_out)


        return x * attention



# =========================================================
# Residual Block
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()


        self.conv1 = DepthwiseSeparableConv(
            channels,
            channels
        )

        self.bn1 = nn.BatchNorm2d(channels)


        self.conv2 = DepthwiseSeparableConv(
            channels,
            channels
        )

        self.bn2 = nn.BatchNorm2d(channels)


        self.cbam = CBAM(channels)



    def forward(self,x):

        identity = x


        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )


        x = self.cbam(x)


        x = self.bn2(
            self.conv2(x)
        )


        x = self.cbam(x)


        x += identity


        return F.relu(x)




# =========================================================
# Teacher Assistant Network
# =========================================================

class TeacherAssistant(nn.Module):

    """
    Teacher Assistant model used in HMT-KD framework.

    It learns from:
    - MobileNetV2 teacher
    - ResNet18 + SENet teacher

    and guides the lightweight student model.
    """

    def __init__(self, num_classes=5):

        super().__init__()



        self.stage0 = nn.Sequential(

            DepthwiseSeparableConv(3,32),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            ResidualBlock(32),

            nn.MaxPool2d(2)

        )



        self.stage1 = nn.Sequential(

            DepthwiseSeparableConv(32,32),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            ResidualBlock(32),

            nn.MaxPool2d(2)

        )



        self.stage2 = nn.Sequential(

            DepthwiseSeparableConv(32,64),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            ResidualBlock(64),

            nn.MaxPool2d(2)

        )



        self.stage3 = nn.Sequential(

            DepthwiseSeparableConv(64,64),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            ResidualBlock(64),

            nn.MaxPool2d(2)

        )



        self.stage4 = nn.Sequential(

            DepthwiseSeparableConv(64,128),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            ResidualBlock(128),

            nn.MaxPool2d(2)

        )



        self.stage5 = nn.Sequential(

            DepthwiseSeparableConv(128,128),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            ResidualBlock(128),

            nn.MaxPool2d(2)

        )



        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)



        self.fc1 = nn.Linear(
            128*2,
            128
        )


        self.bn_fc1 = nn.BatchNorm1d(128)

        self.dropout1 = nn.Dropout(0.5)



        self.fc2 = nn.Linear(
            128,
            64
        )


        self.bn_fc2 = nn.BatchNorm1d(64)

        self.dropout2 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(
            64,
            num_classes
        )



    def forward(self,x):


        x = self.stage0(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = self.stage5(x)



        avg = self.avg_pool(x)

        max_pool = self.max_pool(x)



        x = torch.cat(
            [avg,max_pool],
            dim=1
        )


        x = torch.flatten(
            x,
            start_dim=1
        )


        x = F.relu(
            self.bn_fc1(
                self.fc1(x)
            )
        )


        x = self.dropout1(x)


        x = F.relu(
            self.bn_fc2(
                self.fc2(x)
            )
        )


        x = self.dropout2(x)


        x = self.fc3(x)


        return x




# =========================================================
# Load pretrained Teacher Assistant
# =========================================================

def load_teacher_assistant(
        checkpoint_path,
        num_classes=5,
        device=None
):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )


    model = TeacherAssistant(
        num_classes=num_classes
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    #  state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )


    
    elif isinstance(checkpoint, nn.Module):

        model = checkpoint


    else:

        model.load_state_dict(checkpoint)



    model.to(device)

    model.eval()


    return model
