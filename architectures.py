import torch
import torch.nn as nn
import torch.nn.functional as F


class KKAN_Small(nn.Module):
    """Small KKAN architecture flexible pour KAN/RBF/Wavelet."""

    def __init__(
        self,
        conv_layer_class,  # KAN_Convolutional_Layer, RBFKAN_Convolutional_Layer, WavKAN_Convolutional_Layer
        conv_params,       # dict des params pour la couche conv
        linear_layer_class,  # KANLinear, RBFLinear, WavLinear
        linear_params,       # dict des params pour la couche linéaire
        num_classes=10
    ):
        super().__init__()
        self.conv1 = conv_layer_class(
            in_channels=1,
            out_channels=5,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            **conv_params
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_layer_class(
            in_channels=5,
            out_channels=5,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            **conv_params
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = linear_layer_class(
            in_features=5 * 5 * 5,  # 5 canaux, 5x5 spatial après 2 poolings sans padding
            out_features=num_classes,
            **linear_params
        )
        self.name = f"KKAN (Small)"

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def regularization_loss(self):
        reg_loss = 0
        if hasattr(self.conv1, 'convs'):
            for conv in self.conv1.convs:
                if hasattr(conv, 'conv'):
                    reg_loss += conv.conv.regularization_loss()
        if hasattr(self.conv2, 'convs'):
            for conv in self.conv2.convs:
                if hasattr(conv, 'conv'):
                    reg_loss += conv.conv.regularization_loss()
        if hasattr(self.fc, 'regularization_loss'):
            reg_loss += self.fc.regularization_loss()
        return reg_loss

class KANC_MLP_Medium(nn.Module):
    """
    KANC MLP Medium architecture - fidèle au papier, avec MLP standard en sortie.
    """
    def __init__(
        self,
        conv_layer_class,    # KAN_Convolutional_Layer, RBFKAN_Convolutional_Layer, WavKAN_Convolutional_Layer
        conv_params,         # dict des params pour la couche conv
        num_classes=10
    ):
        super().__init__()
        self.conv1 = conv_layer_class(
            in_channels=1,
            out_channels=5,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            **conv_params
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_layer_class(
            in_channels=5,
            out_channels=10,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            **conv_params
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(10 * 5 * 5, num_classes)  # MLP standard
        self.name = f"KANC MLP (Medium)"

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def regularization_loss(self):
        reg_loss = 0
        if hasattr(self.conv1, 'convs'):
            for conv in self.conv1.convs:
                if hasattr(conv, 'conv'):
                    reg_loss += conv.conv.regularization_loss()
        if hasattr(self.conv2, 'convs'):
            for conv in self.conv2.convs:
                if hasattr(conv, 'conv'):
                    reg_loss += conv.conv.regularization_loss()
        return reg_loss