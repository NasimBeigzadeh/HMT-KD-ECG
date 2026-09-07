# -*- coding: utf-8 -*-

"""
Data augmentation and normalization transforms
for ECG image classification.

Supported datasets:
- PTB-XL
- Chapman
"""

from torchvision import transforms as T


# =====================================================
# PTB-XL transforms
# =====================================================

PTBXL_MEAN = (
    0.9610,
    0.8270,
    0.8528
)

PTBXL_STD = (
    0.1164,
    0.1174,
    0.0648
)


def get_ptbxl_transforms():

    train_transform = T.Compose([
        T.Resize((300, 300)),

        T.RandomHorizontalFlip(
            p=0.7
        ),

        T.RandomRotation(
            0.2
        ),

        T.ColorJitter(
            brightness=0.2,
            contrast=0.3,
            saturation=0.2,
            hue=0.2
        ),

        T.ToTensor(),

        T.Normalize(
            mean=PTBXL_MEAN,
            std=PTBXL_STD
        )
    ])


    valid_transform = T.Compose([
        T.Resize((300,300)),

        T.ToTensor(),

        T.Normalize(
            mean=PTBXL_MEAN,
            std=PTBXL_STD
        )
    ])


    test_transform = T.Compose([
        T.Resize((300,300)),

        T.ToTensor(),

        T.Normalize(
            mean=PTBXL_MEAN,
            std=PTBXL_STD
        )
    ])


    return (
        train_transform,
        valid_transform,
        test_transform
    )



# =====================================================
# Chapman transforms
# =====================================================


CHAPMAN_MEAN = (
    0.9178,
    0.7892,
    0.8450
)


CHAPMAN_STD = (
    0.1427,
    0.1356,
    0.0645
)



def get_chapman_transforms():


    train_transform = T.Compose([

        T.Resize((300,300)),

        T.RandomHorizontalFlip(
            p=0.7
        ),

        T.RandomRotation(
            0.2
        ),

        T.ColorJitter(
            brightness=0.2,
            contrast=0.3,
            saturation=0.2,
            hue=0.2
        ),

        T.ToTensor(),

        T.Normalize(
            mean=CHAPMAN_MEAN,
            std=CHAPMAN_STD
        )
    ])



    valid_transform = T.Compose([

        T.Resize((300,300)),

        T.ToTensor(),

        T.Normalize(
            mean=CHAPMAN_MEAN,
            std=CHAPMAN_STD
        )
    ])



    test_transform = T.Compose([

        T.Resize((300,300)),

        T.ToTensor(),

        T.Normalize(
            mean=CHAPMAN_MEAN,
            std=CHAPMAN_STD
        )
    ])



    return (
        train_transform,
        valid_transform,
        test_transform
    )
