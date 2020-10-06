# -*- coding: utf-8 -*-
import torch
import math

def premute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_flattened.append(premute_and_flatten(box_cls_per_level, N, A, C, H, W))
        box_regression_flattened.append(premute_and_flatten(box_regression_per_level, N, A, 4, H, W))
        
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression, dim=1).reshape(-1, 4)
    return box_cls, box_regression


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def encode_boxes(reference_boxes, proposals, weights=(1.0, 1.0, 1.0, 1.0)):
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
    
    targets_dx = weights[0] * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = weights[1] * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = weights[2] * torch.log(gt_widths / ex_widths)
    targets_dh = weights[3] * torch.log(gt_heights / ex_heights)
    
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


def decode_boxes(proposals, targets, bbox_xform_clip, weights=(1.0, 1.0, 1.0, 1.0)):
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)
    
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    dx = targets[:, 0] / weights[0]
    dy = targets[:, 1] / weights[1]
    dw = torch.clamp(targets[:, 2] / weights[2], max=bbox_xform_clip)
    dh = torch.clamp(targets[:, 3] / weights[3], max=bbox_xform_clip)
    
    pred_ctr_x = dx[:, None] * ex_widths + ex_ctr_x
    pred_ctr_y = dy[:, None] * ex_heights + ex_ctr_y
    pred_widths = torch.exp(dw[:, None]) * ex_widths
    pred_heights = torch.exp(dh[:, None]) * ex_heights
    
    pred_x1 = pred_ctr_x - 0.5 * pred_widths
    pred_x2 = pred_ctr_x + 0.5 * pred_widths
    pred_y1 = pred_ctr_y - 0.5 * pred_heights
    pred_y2 = pred_ctr_y + 0.5 * pred_heights
    
    pred_boxes = torch.cat((pred_x1, pred_y1, pred_x2, pred_y2), dim=1)
    return pred_boxes


class BoxCoder(object):
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
        
        
    def encode(self, reference_boxes, proposals):
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = encode_boxes(reference_boxes, proposals, self.weights)
        
    
    def decode(self, rel_codes, boxes):
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        pred_boxes = decode_boxes(concat_boxes, rel_codes.reshape(box_sum, -1), self.bbox_xform_clip)
        return pred_boxes.reshape(box_sum, -1, 4)
        

class BalancedPositiveNegativeSampler(object):
    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        
    
    def __call__(self, matched_idxs):
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
            
        return pos_idx, neg_idx
    

class Matcher(object):
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches


    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None
            
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS
        
        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches
    
    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


def boxes_intersection(boxes_1, boxes_2):
    A = boxes_1.size(0)
    B = boxes_2.size(0)
    max_xy = torch.min(boxes_1[:, 2:].unsqueeze(1).expand(A, B, 2),
                       boxes_2[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(boxes_1[:, :2].unsqueeze(1).expand(A, B, 2),
                       boxes_2[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def boxes_iou(boxes_1, boxes_2):
    inter = boxes_intersection(boxes_1, boxes_2)
    area_1 = ((boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])).unsqueeze(1).expand_as(inter)
    area_2 = ((boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_1 + area_2 - inter
    return inter / union


def clip_boxes_to_image(boxes, size):
    boxes_x = boxes[:, 0::2]
    boxes_y = boxes[:, 1::2]
    height, width = size
    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=2)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep


if __name__ == '__main__':
    # proposals = torch.Tensor([-1, -1, 1, 1])[None, :] * torch.linspace(11, 20, 10)[:, None]
    # delta = torch.randn(proposals.shape)
    # reference_boxes = proposals + delta
    # print(reference_boxes)
    # targets = encode_boxes(reference_boxes, proposals)
    # print(targets)
    # pred_boxes = decode_boxes(proposals, targets, math.log(1000. / 16))
    # print(pred_boxes)
    # match_quality_matrix = torch.randn([5, 5])
    # print(match_quality_matrix)
    # matcher = Matcher(0.9, -0.5)
    # matches = matcher(match_quality_matrix)
    # print(matches)
    # matcher.allow_low_quality_matches = True
    # matches = matcher(match_quality_matrix)
    # print(matches)
    pass