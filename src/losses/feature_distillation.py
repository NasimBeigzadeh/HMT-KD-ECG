
"""
Feature-based Knowledge Distillation Loss

This module implements intermediate feature transfer from
Teacher Assistant (TA) to Student network.

The student learns from hidden feature representations
extracted from intermediate convolutional layers of the TA.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureHookManager:
    """
    Register forward hooks to extract intermediate features
    from teacher assistant and student networks.
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        teacher_layers,
        student_layers
    ):

        self.teacher_features = {}
        self.student_features = {}
        self.hooks = []


        assert len(teacher_layers) == len(student_layers), \
            "Teacher and student feature layers must have the same length"


        for t_layer, s_layer in zip(
            teacher_layers,
            student_layers
        ):

            self._register_hook(
                teacher_model,
                t_layer,
                self.teacher_features
            )

            self._register_hook(
                student_model,
                s_layer,
                self.student_features
            )



    def _get_layer(self, model, layer_name):

        for name, layer in model.named_modules():

            if name == layer_name:
                return layer


        raise ValueError(
            f"Layer {layer_name} not found in model"
        )



    def _register_hook(
        self,
        model,
        layer_name,
        feature_storage
    ):

        layer = self._get_layer(
            model,
            layer_name
        )


        def hook_function(module, input, output):

            feature_storage[layer_name] = output



        hook = layer.register_forward_hook(
            hook_function
        )

        self.hooks.append(hook)



    def clear(self):

        """
        Clear stored feature maps before each batch.
        """

        self.teacher_features.clear()
        self.student_features.clear()



    def remove_hooks(self):

        """
        Remove all forward hooks.
        """

        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()



def align_feature_maps(
    student_feature,
    teacher_feature
):
    """
    Align spatial and channel dimensions
    between student and teacher features.
    """


    # Resize spatial dimensions

    if student_feature.shape[2:] != teacher_feature.shape[2:]:

        student_feature = F.interpolate(
            student_feature,
            size=teacher_feature.shape[2:],
            mode="bilinear",
            align_corners=False
        )


    # Align channel dimension

    if student_feature.shape[1] != teacher_feature.shape[1]:

        projection = nn.Conv2d(
            student_feature.shape[1],
            teacher_feature.shape[1],
            kernel_size=1,
            bias=False
        ).to(student_feature.device)


        student_feature = projection(
            student_feature
        )


    return student_feature



def feature_distillation_loss(
    teacher_features,
    student_features
):

    """
    Calculate feature-based KD loss.

    Mean Squared Error is used to minimize
    the difference between TA and student
    intermediate representations.

    """

    loss = 0.0

    num_features = len(teacher_features)


    for teacher_key, student_key in zip(
        teacher_features.keys(),
        student_features.keys()
    ):


        teacher_feature = teacher_features[teacher_key].detach()

        student_feature = student_features[student_key]


        student_feature = align_feature_maps(
            student_feature,
            teacher_feature
        )


        loss += F.mse_loss(
            student_feature,
            teacher_feature
        )



    return loss / num_features
