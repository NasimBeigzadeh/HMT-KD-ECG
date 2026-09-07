
import torch
import torch.nn as nn
import torch.nn.functional as F



class DepthwiseSeparableConv(nn.Module):

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
            kernel_size,
            stride,
            padding,
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



class CBAM(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):
        super().__init__()

        reduced_channels = max(
            channels // reduction,
            1
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc1 = nn.Conv2d(
            channels,
            reduced_channels,
            1,
            bias=False
        )


        self.fc2 = nn.Conv2d(
            reduced_channels,
            channels,
            1,
            bias=False
        )


        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

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


        scale = self.sigmoid(
            avg_out + max_out
        )


        return x * scale




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



    def forward(self, x):

        residual = x


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


        x += residual


        return F.relu(x)




class StudentCNN(nn.Module):

    def __init__(
        self,
        num_classes=5
    ):
        super().__init__()



        self.conv0 = DepthwiseSeparableConv(
            3,
            32
        )

        self.bn0 = nn.BatchNorm2d(32)

        self.pool0 = nn.MaxPool2d(
            2,
            2
        )

        self.res0 = ResidualBlock(32)



        self.conv1 = DepthwiseSeparableConv(
            32,
            32
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(
            2,
            2
        )

        self.res1 = ResidualBlock(32)




        self.conv2 = DepthwiseSeparableConv(
            32,
            48
        )

        self.bn2 = nn.BatchNorm2d(48)

        self.pool2 = nn.MaxPool2d(
            2,
            2
        )

        self.res2 = ResidualBlock(48)




        self.conv22 = DepthwiseSeparableConv(
            48,
            48
        )

        self.bn22 = nn.BatchNorm2d(48)

        self.pool22 = nn.MaxPool2d(
            2,
            2
        )

        self.res22 = ResidualBlock(48)




        self.conv3 = DepthwiseSeparableConv(
            48,
            64
        )

        self.bn3 = nn.BatchNorm2d(64)

        self.pool3 = nn.MaxPool2d(
            2,
            2
        )

        self.res3 = ResidualBlock(64)




        self.conv33 = DepthwiseSeparableConv(
            64,
            64
        )

        self.bn33 = nn.BatchNorm2d(64)

        self.pool33 = nn.MaxPool2d(
            2,
            2
        )

        self.res33 = ResidualBlock(64)



        self.global_pool = nn.AdaptiveAvgPool2d(1)



        self.fc1 = nn.Linear(
            64,
            128
        )

        self.bn_fc = nn.BatchNorm1d(128)

        self.dropout1 = nn.Dropout(0.5)



        self.fc2 = nn.Linear(
            128,
            256
        )

        self.bn_fc2 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(
            256,
            num_classes
        )



    def forward(self, x):

        x = F.relu(
            self.bn0(
                self.conv0(x)
            )
        )

        x = self.pool0(x)

        x = self.res0(x)



        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.pool1(x)

        x = self.res1(x)



        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = self.pool2(x)

        x = self.res2(x)



        x = F.relu(
            self.bn22(
                self.conv22(x)
            )
        )

        x = self.pool22(x)

        x = self.res22(x)



        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = self.pool3(x)

        x = self.res3(x)



        x = F.relu(
            self.bn33(
                self.conv33(x)
            )
        )

        x = self.pool33(x)

        x = self.res33(x)



        x = self.global_pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )



        x = F.relu(
            self.bn_fc(
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




def load_student(
        checkpoint_path=None,
        num_classes=5,
        device=None
):

    if device is None:

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )


    model = StudentCNN(
        num_classes=num_classes
    )


    if checkpoint_path is not None:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )


        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            model.load_state_dict(checkpoint)



    model.to(device)


    return model
