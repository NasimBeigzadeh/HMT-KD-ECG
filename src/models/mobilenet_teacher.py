
"""
MobileNetV2 Teacher Network

This module defines the MobileNetV2 teacher model
used in the HMT-KD framework.

The teacher network is fine-tuned on ECG image
classification and provides high-level soft targets
for knowledge distillation.
"""


import torch
import torch.nn as nn
from torchvision import models




def create_mobilenet_teacher(
        num_classes=5,
        pretrained_weights=None,
        device=None
):

    """
    Create MobileNetV2 teacher model.

    Args:
        num_classes:
            Number of ECG classes.

        pretrained_weights:
            Path to trained teacher checkpoint.

        device:
            cpu or cuda.


    Returns:
        MobileNetV2 teacher model
    """



    if device is None:

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )



    # ------------------------------------------------
    # MobileNetV2 backbone
    # ------------------------------------------------

    model = models.mobilenet_v2(
        weights=None
    )



    # ------------------------------------------------
    # Freeze backbone
    # ------------------------------------------------

    for param in model.parameters():

        param.requires_grad = False



    # ------------------------------------------------
    # Fine tuning selected layers
    # Same strategy used in experiments
    # ------------------------------------------------

    trainable_indices = [

        0,1,2,3,4,5,
        6,7,8,9,10,
        11,12

    ]



    for idx, layer in enumerate(
        model.features
    ):


        if idx in trainable_indices:


            for param in layer.parameters():

                param.requires_grad = True




    # ------------------------------------------------
    # Replace classifier
    # ------------------------------------------------

    model.classifier[1] = nn.Linear(

        model.classifier[1].in_features,

        num_classes

    )




    # ------------------------------------------------
    # Load pretrained teacher weights
    # ------------------------------------------------

    if pretrained_weights is not None:


        checkpoint = torch.load(

            pretrained_weights,

            map_location=device

        )


        model.load_state_dict(
            checkpoint
        )




    model.to(device)



    return model





# ------------------------------------------------
# Parameter information
# ------------------------------------------------

def model_info(model):


    total_params = sum(

        p.numel()

        for p in model.parameters()

    )


    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


    print(
        f"Total parameters: {total_params:,}"
    )


    print(
        f"Trainable parameters: {trainable_params:,}"
    )
