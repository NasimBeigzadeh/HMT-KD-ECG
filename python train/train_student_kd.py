
"""
Training pipeline for HMT-KD framework.

This module implements hierarchical multi-teacher
knowledge distillation training with:
1. Response-based KD from MobileNetV2 and ResNet18+SENet teachers
2. Feature-based KD from Teacher Assistant
3. Supervised learning from ground-truth labels
"""


import os
import gc
import torch

import torch.nn as nn
import torch.optim as optim

from torch.amp import autocast, GradScaler
from torchmetrics import Accuracy
from tqdm import tqdm


from src.losses.knowledge_distillation import (
    compute_distillation_loss,
    adaptive_weights,
    weighted_teacher_outputs
)


from src.losses.feature_distillation import (
    FeatureHookManager,
    feature_distillation_loss
)



# -------------------------------------------------
# Device configuration
# -------------------------------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# -------------------------------------------------
# Average Meter
# -------------------------------------------------

class AverageMeter:

    def __init__(self):
        self.reset()


    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(
        self,
        value,
        n=1
    ):

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count




# -------------------------------------------------
# Validation
# -------------------------------------------------

def validate(
    model,
    loader,
    loss_function,
    num_classes=5
):

    model.eval()

    loss_meter = AverageMeter()

    accuracy = Accuracy(
        task="multiclass",
        num_classes=num_classes
    ).to(device)



    with torch.no_grad():

        for images, labels in loader:


            images = images.to(device)
            labels = labels.to(device)


            with autocast(device_type="cuda"):

                outputs = model(images)

                loss = loss_function(
                    outputs,
                    labels
                )


            loss_meter.update(
                loss.item(),
                images.size(0)
            )


            accuracy.update(
                outputs,
                labels
            )



    return (
        loss_meter.avg,
        accuracy.compute().item()
    )





# -------------------------------------------------
# One Epoch HMT-KD Training
# -------------------------------------------------

def train_one_epoch(
        student,
        teachers,
        feature_manager,
        loader,
        optimizer,
        scaler,
        teacher_scores,
        temperature=2,
        alpha=0.7,
        beta=0.2,
        epoch=0
):


    student.train()


    loss_meter = AverageMeter()


    accuracy = Accuracy(
        task="multiclass",
        num_classes=5
    ).to(device)



    teacher_weights = adaptive_weights(
        teacher_scores
    )



    progress = tqdm(
        loader,
        desc=f"Epoch {epoch+1}",
        unit="batch"
    )



    for images, labels in progress:


        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        feature_manager.clear()



        with autocast(device_type="cuda"):



            # Student prediction

            student_output = student(
                images
            )



            # Teacher prediction

            with torch.no_grad():

                teacher_outputs = [

                    teacher(images)

                    for teacher in teachers[:2]

                ]


                # TA forward pass
                teachers[2](images)



            # Ensemble teacher knowledge

            combined_teacher = weighted_teacher_outputs(

                teacher_outputs,
                teacher_weights

            )



            # Response KD

            response_loss = compute_distillation_loss(

                student_output,
                labels,
                combined_teacher,
                temperature,
                alpha

            )



            # Feature KD

            feature_loss = feature_distillation_loss(

                feature_manager.teacher_features,
                feature_manager.student_features

            )



            # Total HMT-KD loss

            loss = (

                (1-beta) * response_loss

                +

                beta * feature_loss

            )




        scaler.scale(loss).backward()

        scaler.step(
            optimizer
        )

        scaler.update()



        loss_meter.update(

            loss.item(),

            images.size(0)

        )


        accuracy.update(

            student_output,

            labels

        )



        progress.set_postfix(

            loss=f"{loss_meter.avg:.4f}",

            acc=f"{accuracy.compute().item()*100:.2f}%"

        )



        del images
        del labels
        del student_output

        torch.cuda.empty_cache()

        gc.collect()



    return (

        loss_meter.avg,

        accuracy.compute().item()

    )





# -------------------------------------------------
# Main Training Function
# -------------------------------------------------

def train_hmt_kd(

        student,

        teachers,

        train_loader,

        valid_loader,

        teacher_layers,

        student_layers,

        epochs=65

):



    student.to(device)


    for teacher in teachers:

        teacher.to(device)

        teacher.eval()



    # Feature transfer manager
    feature_manager = FeatureHookManager(

        teachers[2],

        student,

        teacher_layers,

        student_layers

    )



    optimizer = optim.AdamW(

        filter(

            lambda p:p.requires_grad,

            student.parameters()

        ),

        lr=4e-4,

        weight_decay=1e-5

    )



    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.85,

        patience=3

    )


    scaler = GradScaler(
        device="cuda"
    )



    criterion = nn.CrossEntropyLoss()



    best_loss = float("inf")



    history = {

        "train_loss":[],
        "train_acc":[],
        "val_loss":[],
        "val_acc":[]

    }



    teacher_scores = [

        0.9538,   # MobileNetV2

        0.9518,   # ResNet18+SENet

        0.9565    # TA

    ]




    for epoch in range(epochs):


        train_loss, train_acc = train_one_epoch(

            student,

            teachers,

            feature_manager,

            train_loader,

            optimizer,

            scaler,

            teacher_scores,

            temperature=2,

            alpha=0.7,

            beta=0.2,

            epoch=epoch

        )



        val_loss, val_acc = validate(

            student,

            valid_loader,

            criterion

        )



        history["train_loss"].append(train_loss)

        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)

        history["val_acc"].append(val_acc)




        if val_loss < best_loss:


            best_loss = val_loss


            torch.save(

                student.state_dict(),

                "STUDENT.pt"

            )



        scheduler.step(
            val_loss
        )



        print(
            f"""
Epoch [{epoch+1}/{epochs}]

Train Loss:
{train_loss:.4f}

Train Accuracy:
{train_acc*100:.2f}%

Validation Loss:
{val_loss:.4f}

Validation Accuracy:
{val_acc*100:.2f}%

"""
        )



    feature_manager.remove_hooks()



    print(
        "Training completed. Best student model saved."
    )



    return history
