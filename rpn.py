# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
import utils
from torchvision.ops import boxes as box_ops

class RpnHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RpnHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        
        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
            
    
    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
    
    
class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = utils.BoxCoder(weight=(1.0, 1.0, 1.0, 1.0))
        self.proposal_matcher = utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        self.fg_bg_sampler = utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3
        
        
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']


    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']
        
        
    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = utils.boxes_iou(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
                
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        
        return labels, matched_gt_boxes
    
    
    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)
            
    
    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
   
        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)
            
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        
        final_boxes = []
        final_scores = []
        
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = utils.clip_boxes_to_image(boxes, img_shape)
            keep = utils.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores
    
    
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness = objectness.flatten()
        
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)    
        
        box_loss = utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())
        
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
        
        return objectness_loss, box_loss
    
    def forward(self, image_sizes, features, targets=None):
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator.gen_anchors(image_sizes, features)
        
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = utils.concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses
        
            
            
            
            
            
            
            
            
            
            
            
            
        
        