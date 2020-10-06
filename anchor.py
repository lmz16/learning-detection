# -*- coding: utf-8 -*-
import torch
from torch import nn

class AnchorGenerator(object):
    def __init__(self, sizes=(128, 256, 512), ratio_aspects=(0.5, 1.0, 2.0), dtype=torch.float32, device='cpu'):
        self.sizes = sizes
        self.ratio_aspects = ratio_aspects
        self.dtype = dtype
        self.device = device
        self.base_anchors = None
        
    
    def gen_base_anchor(self, sizes=None, ratio_aspects=None):
        if (sizes == None) or (ratio_aspects == None):
            sizes = self.sizes
            ratio_aspects = self.ratio_aspects
            
        assert sizes is not None
        assert ratio_aspects is not None
        
        sizes = torch.as_tensor(sizes, dtype=self.dtype, device=self.device)
        ratio_aspects = torch.as_tensor(ratio_aspects, dtype=self.dtype, device=self.device)
        
        w_ratios = torch.sqrt(ratio_aspects)
        h_ratios = 1 / w_ratios
        
        ws = (w_ratios[None, :] * sizes[:, None]).view(-1)
        hs = (h_ratios[None, :] * sizes[:, None]).view(-1)
        
        self.base_anchors = (torch.stack([-ws, -hs, ws, hs], dim=1) / 2).round()
    
    
    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        assert self.base_anchors is not None
        
        for grid_size, stride in zip(grid_sizes, strides):
            grid_height, grid_width = grid_size
            stride_height, stride_width = stride
            
            shifts_x = torch.arange(0, grid_width, dtype=self.dtype, device=self.device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=self.dtype, device=self.device) * stride_height
            
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            
            anchors.append((self.base_anchors.view(-1, 1, 4) + shifts.view(1, -1, 4)).reshape(-1, 4))
            
        return anchors
    
    
    def gen_anchors(self, image_sizes, feature_maps):
        assert len(image_sizes) == len(feature_maps)
        grid_sizes = [list(feature_map.shape[-2:]) for feature_map in feature_maps]
        strides = [[image_sizes[i][0] // grid_sizes[i][0], 
                    image_sizes[i][1] // grid_sizes[i][1]]
                   for i in range(len(image_sizes))]
        return self.grid_anchors(grid_sizes, strides)


if __name__ == '__main__':
    ag = AnchorGenerator()
    ag.gen_base_anchor()
    image_sizes = [(100, 100) for _ in range(5)]
    feature_maps = [torch.Tensor(5, 5, 5) for _ in range(5)]
    print(ag.gen_anchors(image_sizes, feature_maps))