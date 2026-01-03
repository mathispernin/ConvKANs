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
        num_classes=10,
        in_channels=1,
        input_size=(28, 28),
        use_batchnorm=False
    ):
        super().__init__()
        # conv/pool settings (same as before)
        k_h, k_w = 3, 3
        s_h, s_w = 1, 1
        p_h, p_w = 0, 0

        self.conv1 = conv_layer_class(
            in_channels=in_channels,
            out_channels=5,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            **conv_params
        )
        self.bn1 = nn.BatchNorm2d(5) if use_batchnorm else None
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_layer_class(
            in_channels=5,
            out_channels=5,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            **conv_params
        )
        self.bn2 = nn.BatchNorm2d(5) if use_batchnorm else None
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # compute output spatial dimensions after conv/pool layers
        def conv_out(dim, kernel, pad, stride):
            return (dim + 2 * pad - kernel) // stride + 1

        h, w = input_size
        h = conv_out(h, k_h, p_h, s_h)
        w = conv_out(w, k_w, p_w, s_w)
        h = h // 2  # pool1
        w = w // 2
        h = conv_out(h, k_h, p_h, s_h)
        w = conv_out(w, k_w, p_w, s_w)
        h = h // 2  # pool2
        w = w // 2

        in_features = 5 * h * w
        self.fc = linear_layer_class(
            in_features=in_features,
            out_features=num_classes,
            **linear_params
        )
        self.name = f"KKAN (Small)"

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
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
        num_classes=10,
        in_channels=1,
        input_size=(28, 28),
        use_batchnorm=False
    ):
        super().__init__()
        k_h, k_w = 3, 3
        s_h, s_w = 1, 1
        p_h, p_w = 0, 0

        self.conv1 = conv_layer_class(
            in_channels=in_channels,
            out_channels=5,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            **conv_params
        )
        self.bn1 = nn.BatchNorm2d(5) if use_batchnorm else None
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_layer_class(
            in_channels=5,
            out_channels=10,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            **conv_params
        )
        self.bn2 = nn.BatchNorm2d(10) if use_batchnorm else None
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # compute final spatial dims
        def conv_out(dim, kernel, pad, stride):
            return (dim + 2 * pad - kernel) // stride + 1

        h, w = input_size
        h = conv_out(h, k_h, p_h, s_h)
        w = conv_out(w, k_w, p_w, s_w)
        h = h // 2
        w = w // 2
        h = conv_out(h, k_h, p_h, s_h)
        w = conv_out(w, k_w, p_w, s_w)
        h = h // 2
        w = w // 2

        in_features = 10 * h * w
        self.fc = nn.Linear(in_features, num_classes)  # MLP standard
        self.name = f"KANC MLP (Medium)"

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
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