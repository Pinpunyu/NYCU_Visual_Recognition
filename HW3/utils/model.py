import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_Weights,  MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


class MaskRCNN(nn.Module):

    def __init__(self, num_classes = 5, backbone = "maskrcnn_resnet50_fpn_v2", backbone_layers=3):
        super().__init__()

        # backbone
        if backbone == "maskrcnn_resnet50_fpn":
            self.model = maskrcnn_resnet50_fpn(weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers = backbone_layers)
        elif backbone == "maskrcnn_resnet50_fpn_v2":
            self.model = maskrcnn_resnet50_fpn_v2(weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, trainable_backbone_layers = backbone_layers)

        # anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        # anchor_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # anchor_generator = AnchorGenerator(sizes=anchor_sizes,aspect_ratios=anchor_ratios)
        # self.model.rpn.anchor_generator = anchor_generator

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # mask head
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,  dim_reduced=256, num_classes=num_classes)

    def forward(self, images, targets=None):  
        return self.model(images, targets)

