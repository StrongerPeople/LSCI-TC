import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

class Weight_soft_CEloss(nn.Module):
    def __init__(self,reduction='mean',imagegamma=1.0,textgamma=1.0,alpha=0.3,maxgamma=2.0,mingamma = -1.0):
        super(Weight_soft_CEloss, self).__init__()
        self.reduction = reduction
        self.imagegamma =  imagegamma
        self.textgamma =  textgamma
        self.maxgamma = torch.tensor(maxgamma)
        self.mingamma = torch.tensor(mingamma)
        self.image_to_textgamma = torch.tensor(imagegamma)
        self.text_to_imagegamma = torch.tensor(textgamma)
        self.alpha = alpha
        self.update = lambda g, gap: self.update_gamma(g, gap, self.maxgamma, self.mingamma)


    def update_gamma(self,gamma, gap, maxgamma, mingamma, max_change=0.2):
        sign = 1 if gamma >= 0 else -1
        exp_factor = torch.exp(sign * gap)
        target = gamma * exp_factor
        delta = target - gamma
        if delta.abs() > max_change: # smooth change
            delta = torch.sign(delta) * max_change
        new_gamma = gamma + delta
        if new_gamma >= 0:
            return min(maxgamma, new_gamma)
        else:
            return max(mingamma, new_gamma)
    
    def updategamma(self,image_text_meancalibration_gap,text_image_meancalibration_gap):
        image_text_meancalibration_gap = torch.tensor(image_text_meancalibration_gap)
        text_image_meancalibration_gap = torch.tensor(text_image_meancalibration_gap)
        self.image_to_textgamma = self.update(self.image_to_textgamma, image_text_meancalibration_gap)
        self.text_to_imagegamma = self.update(self.text_to_imagegamma, text_image_meancalibration_gap)
        # inv-focal loss - focal loss switch
        if abs(self.image_to_textgamma) <  abs(self.alpha):
            self.image_to_textgamma = - torch.tensor(self.alpha * self.image_to_textgamma)
        elif abs(self.text_to_imagegamma) <  abs(self.alpha):
            self.text_to_imagegamma = - torch.tensor(self.alpha * self.text_to_imagegamma)
        print(f"update image_to_textgamma to {self.image_to_textgamma}")
        print(f"update text_to_imagegamma to {self.text_to_imagegamma}")
        return 0
    
    def forward(self, inputs, labels,text_to_image=False):
        """"
            warning: the weight shape should be same with the inputs shape,the bug 0**0 
        """
        pt = F.softmax(inputs, dim=1)
        if text_to_image:
            gamma_sign = torch.sign(self.text_to_imagegamma)
            pt = gamma_sign * pt
            gamma_mag = abs(self.text_to_imagegamma)
            weight_log_softmax = torch.clamp(abs(labels - pt),min=1e-5,max=2)**gamma_mag * labels
        else:
            gamma_sign = torch.sign(self.image_to_textgamma)
            pt = gamma_sign * pt
            gamma_mag = abs(self.image_to_textgamma)
            weight_log_softmax = torch.clamp(abs(labels - pt),min=1e-5,max=2)**gamma_mag * labels
        loss = torch._C._nn.cross_entropy_loss(
                inputs,
                weight_log_softmax
        ).mean()
        return loss
