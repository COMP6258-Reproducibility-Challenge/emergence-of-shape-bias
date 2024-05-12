import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    BasicBlock represents a fundamental unit within smaller ResNet architectures, such as ResNet-18 and ResNet-34.
    The block is caracterised by two convolutional layers, each followed by a batch normalisation layer.
    The downsample parameter is used to match the dimensions of the input and output tensors.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int): The stride of the first convolutional layer.
        downsample (nn.Module): The downsample operation to match the dimensions of the input and output tensors.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        # The shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        output += identity

        output = self.relu(output)

        return output


class ResNet(nn.Module):
    """
    The ResNet class defines the architecture of the ResNet model.

    Args:
        basic_block (nn.Module): The basic block used in the ResNet architecture.
        num_blocks (list): The number of blocks in each layer of the ResNet architecture.
        num_classes (int): The number of classes in the dataset.
        model_spec (str): The model specification. CS for Code-Spec and PS for Paper-Spec.
        topk_operation (str): The top-k operation to apply to the model.
        device (str): The device to use for the model.
    """

    def __init__(
        self,
        basic_block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=10,
        model_spec="CS",
        topk_operation=None,
        device="cuda",
    ):
        super(ResNet, self).__init__()
        self.model_spec = model_spec

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # The ResNet architecture is composed of four blocks, each containing a number of basic blocks
        self.block1 = self._make_layer(basic_block, 64, num_blocks[0], stride=1)
        self.block2 = self._make_layer(basic_block, 128, num_blocks[1], stride=2)
        self.block3 = self._make_layer(basic_block, 256, num_blocks[2], stride=2)
        self.block4 = self._make_layer(basic_block, 512, num_blocks[3], stride=2)

        if model_spec == "CS":
            self.fc = nn.Linear(512 * 7 * 7, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)

        self.device = device
        self.topk_operation = topk_operation

    def top_k(self, x, topk_ratio=0.2):
        """
        The implementation of the top-k operation, which retains only the top-k% of the neurons with the highest
        absolute values in the tensor.

        Args:
            x (torch.Tensor): The input tensor.
            topk_ratio (float): The ratio of the top-k neurons to retain.
        """
        n, c, h, w = x.size()
        x_reshape = x.view(n, c, h * w)
        num_kept_neurons = int(topk_ratio * h * w)
        _, indices = torch.topk(x_reshape.abs(), num_kept_neurons, dim=2)
        top_k_mask = torch.zeros_like(x_reshape).scatter_(2, indices, 1).to(self.device)
        sparse_x_reshape = x_reshape * top_k_mask

        # Replace the top-k values with the mean of the top-k values
        if self.topk_operation == "top_k_mean_replace":
            means_topk = sparse_x_reshape.sum(dim=2) / num_kept_neurons
            non_zero_mask = sparse_x_reshape != 0
            means_expanded = means_topk.unsqueeze(2).expand_as(sparse_x_reshape)
            sparse_x_reshape = torch.where(non_zero_mask, means_expanded, sparse_x_reshape).to(self.device)

        sparse_x = sparse_x_reshape.view(n, c, h, w)
        return sparse_x

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """
        Used to create the BasicBlock layers for the ResNet architecture.

        Args:
            block (nn.Module): The block to use in the layer.
            out_channels (int): The number of output channels.
            num_blocks (int): The number of blocks in the layer.
            stride (int): The stride of the first convolutional layer.
        """
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.block1(output)
        output = self.block2(output)

        if self.topk_operation:
            output = self.top_k(output)

        output = self.block3(output)
        output = self.block4(output)

        if self.model_spec == "CS":
            output = F.avg_pool2d(output, 2)
        else:
            output = F.adaptive_avg_pool2d(output, (1, 1))

        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
