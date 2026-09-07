# -*- coding: utf-8 -*-

"""
Dataset loader for ECG image classification.

Supports:
- PTB-XL
- Chapman
"""


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



def create_dataloader(
        train_path,
        valid_path,
        test_path,
        train_transform,
        valid_transform,
        test_transform,
        batch_size=128
):


    train_dataset = ImageFolder(
        train_path,
        transform=train_transform
    )


    valid_dataset = ImageFolder(
        valid_path,
        transform=valid_transform
    )


    test_dataset = ImageFolder(
        test_path,
        transform=test_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return (
        train_loader,
        valid_loader,
        test_loader
    )
