from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def get_model(pretrained, model, num_classes):
    weights = {"resnet18": ResNet18_Weights,
               "resnet50": ResNet50_Weights,
               "ViT": ViT_B_16_Weights}
    tokenizer = None
    if pretrained:
        weights = weights[model]
    else:
        weights = None

    if model == 'resnet18':
        net = resnet18(weights=weights)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif model == 'resnet50':
        net = resnet50(weights=weights)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif model == "ViT":
        net = vit_b_16(weights=weights)
        net.heads[0] = torch.nn.Linear(net.heads[0].in_features, num_classes)
        # Have to replace _scaled_dot_product_efficient_attention with CustomScaledDotProductAttention
        for block in net.encoder.layers:
            original_attention = block.attn

            # Initialize custom attention with the same dimensions
            custom_attention = CustomScaledDotProductAttention(
                embed_dim=original_attention.qkv.in_features,
                num_heads=original_attention.num_heads,
                dropout=original_attention.proj_drop.p
            )

            # Copy weights from the original attention layer to the custom one
            custom_attention.qkv_proj.weight.data = original_attention.qkv.weight.data.clone()
            custom_attention.qkv_proj.bias.data = original_attention.qkv.bias.data.clone()
            custom_attention.out_proj.weight.data = original_attention.proj.weight.data.clone()
            custom_attention.out_proj.bias.data = original_attention.proj.bias.data.clone()

            # Replace the attention layer in the block
            block.attn = custom_attention

    elif model == "BERT":
        # Do not support non-pretrained BERT since it does not make sense
        net = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    else:
        print('choose a valid model - resnet18, resnet50, ViT, BERT', flush=True)
        raise ValueError

    return net, tokenizer

class CustomScaledDotProductAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = torch.nn.Linear(embed_dim, embed_dim * 3)  # For query, key, value
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)      # Output projection
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)

# # https://github.com/lolemacs/continuous-sparsification/blob/master/models/layers.py
# class CustomConv(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#         padding_mode='zeros',device=None, dtype=None):
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
#         padding_mode,device, dtype)
#     def forward(self, input):
#         return self._conv_forward(input, self.weight, self.bias)
#
#     def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
#         return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
#
#     def meta_forward(self, input, fast_weights, name):
#         return self._conv_forward(input, fast_weights[name+".weight"], self.bias) # We just use self.bias since all the convolutions have bias=False
#
# class CustomFC(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
#         super().__init__(in_features, out_features, bias, device, dtype)
#
#     def forward(self, input):
#         return F.linear(input, self.weight, self.bias)
#
#     def meta_forward(self, input, fast_weights, name):
#         return F.linear(input, fast_weights[name+".weight"], fast_weights[name+".bias"])
#
#
# class CustomBN(nn.BatchNorm2d):
#     def __init__(self, inplanes):
#         super().__init__(inplanes)
#
#     # Code from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d, replace self.weight and self.bias with fast_weights
#     def meta_forward(self, input, fast_weights, name):
#         self._check_input_dim(input)
#
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:  # type: ignore[has-type]
#                 self.num_batches_tracked.add_(1)  # type: ignore[has-type]
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)
#
#         return F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean
#             if not self.training or self.track_running_stats
#             else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             fast_weights[name+".weight"],
#             fast_weights[name+".bias"],
#             bn_training,
#             exponential_average_factor,
#             self.eps,
#         )
#
# # https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
#
# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
#     """3x3 convolution with padding"""
#     return CustomConv(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#     )
#
#
# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return CustomConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion: int = 1
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = CustomBN
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#     def meta_forward(self, x, fast_weights, name):
#         identity = x
#
#         out = self.conv1.meta_forward(x, fast_weights, name+".conv1")
#         out = self.bn1.meta_forward(out, fast_weights, name+".bn1")
#         out = self.relu(out)
#
#         out = self.conv2.meta_forward(out, fast_weights, name+".conv2")
#         out = self.bn2.meta_forward(out, fast_weights, name+".bn2")
#
#         if self.downsample is not None:
#             identity = self.downsample[1].meta_forward(self.downsample[0].meta_forward(x, fast_weights, name+".downsample.0"), fast_weights, name+".downsample.1")
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#
#     expansion: int = 4
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = CustomBN
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#     def meta_forward(self, x, fast_weights, name):
#         identity = x
#
#         out = self.conv1.meta_forward(x, fast_weights, name+".conv1")
#         out = self.bn1.meta_forward(out, fast_weights, name+".bn1")
#         out = self.relu(out)
#
#         out = self.conv2.meta_forward(out, fast_weights, name+".conv2")
#         out = self.bn2.meta_forward(out, fast_weights, name+".bn2")
#         out = self.relu(out)
#
#         out = self.conv3.meta_forward(out, fast_weights, name+".conv3")
#         out = self.bn3.meta_forward(out, fast_weights, name+".bn3")
#
#         if self.downsample is not None:
#             identity = self.downsample[1].meta_forward(self.downsample[0].meta_forward(x, fast_weights, name+".downsample.0"), fast_weights, name+".downsample.1")
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#     def __init__(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = CustomBN
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = CustomConv(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = CustomFC(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
#
#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#     ) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
#
#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)
#
#     def meta_forward(self, x, fast_weights):
#         x = self.conv1.meta_forward(x, fast_weights, "conv1")
#         x = self.bn1.meta_forward(x, fast_weights, "bn1")
#         x = self.relu(x)
#         x = self.maxpool(x)
#         for idx, block in enumerate(self.layer1):
#             x = block.meta_forward(x, fast_weights, f"layer1.{idx}")
#         for idx, block in enumerate(self.layer2):
#             x = block.meta_forward(x, fast_weights, f"layer2.{idx}")
#         for idx, block in enumerate(self.layer3):
#             x = block.meta_forward(x, fast_weights, f"layer3.{idx}")
#         for idx, block in enumerate(self.layer4):
#             x = block.meta_forward(x, fast_weights, f"layer4.{idx}")
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc.meta_forward(x, fast_weights, "fc")
#         return x
#
#
# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights,
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#
#     model = ResNet(block, layers, **kwargs)
#
#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
#
#     return model
#
#
# def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet18_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet18_Weights
#         :members:
#     """
#     weights = ResNet18_Weights.verify(weights)
#
#     return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
#
#
#
# def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet34_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet34_Weights
#         :members:
#     """
#     weights = ResNet34_Weights.verify(weights)
#
#     return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
#
#
# def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
#
#     .. note::
#        The bottleneck of TorchVision places the stride for downsampling to the second 3x3
#        convolution while the original paper places it to the first 1x1 convolution.
#        This variant improves the accuracy and is known as `ResNet V1.5
#        <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet50_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet50_Weights
#         :members:
#     """
#     weights = ResNet50_Weights.verify(weights)
#
#     return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
#
#
#
# def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
#
#     .. note::
#        The bottleneck of TorchVision places the stride for downsampling to the second 3x3
#        convolution while the original paper places it to the first 1x1 convolution.
#        This variant improves the accuracy and is known as `ResNet V1.5
#        <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet101_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet101_Weights
#         :members:
#     """
#     weights = ResNet101_Weights.verify(weights)
#
#     return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
#
#
# def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
#
#     .. note::
#        The bottleneck of TorchVision places the stride for downsampling to the second 3x3
#        convolution while the original paper places it to the first 1x1 convolution.
#        This variant improves the accuracy and is known as `ResNet V1.5
#        <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
#
#     Args:
#         weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet152_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ResNet152_Weights
#         :members:
#     """
#     weights = ResNet152_Weights.verify(weights)
#
#     return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)
