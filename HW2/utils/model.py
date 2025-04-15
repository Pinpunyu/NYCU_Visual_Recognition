import torchvision
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import resnext50_32x4d
from torchvision.models.detection.backbone_utils import BackboneWithFPN

class CustomPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

class Faster_RCNN:
    def __init__(self, num_classes=11, backbone='resnet50_fpn', pretrained=True, freeze_backbone=False, freeze_rpn=False, use_custom_head=False):
 
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.freeze_rpn = freeze_rpn
        self.use_custom_head = use_custom_head

        self.model = self._build_model()

    def _build_anchor_generator(self):

        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        anchor_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)


        return AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_ratios
        )


    def _build_model(self):
        if self.backbone_name == 'resnet50_fpn':
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        elif self.backbone_name == 'resnet50_fpn_v2':
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        elif self.backbone_name == 'resnext50_fpn':
            backbone = resnext50_32x4d(weights="DEFAULT" if self.pretrained else None)

            # Remove classification head
            modules = list(backbone.children())[:-2]  # keep up to layer4
            backbone.body = nn.Sequential(*modules)

            # Build FPN from selected layers
            # Extract layers by name based on torchvision's naming convention
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3',}

            in_channels_list = [256, 512, 1024, 2048]
            out_channels = 256

            fpn_backbone = BackboneWithFPN(backbone, return_layers=return_layers, in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=None)
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )

            anchor_generator = self._build_anchor_generator()
            model = FasterRCNN(
                backbone=fpn_backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )

            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Replace anchor generator with smaller anchors
        anchor_generator = self._build_anchor_generator()
        model.rpn.anchor_generator = anchor_generator
        
        # Replace the box predictor head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        if self.use_custom_head:
            model.roi_heads.box_predictor = CustomPredictor(in_features, self.num_classes)
        else:
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Optionally freeze parts of the network
        if self.freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False

        if self.freeze_rpn:
            for param in model.rpn.parameters():
                param.requires_grad = False

        return model
