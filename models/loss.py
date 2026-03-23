import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Weight_soft_CEloss(nn.Module):
    def __init__(self,reduction='mean',imagegamma=1.0,textgamma=0.0,alpha=0.1,maxgamma=5.0,mingamma = -1.0):
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

    def _init_tasks_gamma(self, image_to_text_gap, text_to_image_gap):
        self.image_to_textgamma = torch.tensor(image_to_text_gap)
        self.text_to_imagegamma = torch.tensor(text_to_image_gap)
        print(f"init image_to_textgamma to {self.image_to_textgamma}")
        print(f"init text_to_imagegamma to {self.text_to_imagegamma}")
        
    
    def update_gamma(self,gamma, gap, maxgamma, mingamma, max_change=0.2):
        sign = 1 if gamma >= 0 else -1
        exp_factor = torch.exp(sign * gap)
        target = gamma * exp_factor
        if target >= 0:
            return min(maxgamma, target)
        else:
            return max(mingamma, target)
    
    def updategamma(self,image_text_meancalibration_gap,text_image_meancalibration_gap):
        image_text_meancalibration_gap = torch.tensor(image_text_meancalibration_gap)
        text_image_meancalibration_gap = torch.tensor(text_image_meancalibration_gap)
        self.image_to_textgamma = self.update(self.image_to_textgamma, image_text_meancalibration_gap)
        self.text_to_imagegamma = self.update(self.text_to_imagegamma, text_image_meancalibration_gap)
        # inv-focal loss - focal loss switch
        if abs(self.image_to_textgamma) <  abs(self.alpha * self.imagegamma):
            self.image_to_textgamma = - torch.tensor(self.alpha * self.imagegamma)
        elif abs(self.text_to_imagegamma) <  abs(self.alpha * self.textgamma):
            self.text_to_imagegamma = - torch.tensor(self.alpha * self.textgamma)
        return 0

    def print_gamma(self):
        print(f"update image_to_textgamma to {self.image_to_textgamma}")
        print(f"update text_to_imagegamma to {self.text_to_imagegamma}")
    
    def forward(self, inputs, labels,text_to_image=False):
        """"
            warning: the weight shape should be same with the loss shape,the bug 0**0 
        """
        loss = nn.CrossEntropyLoss()(inputs, labels)
        log_inputs = F.log_softmax(inputs,dim=1)
        pt = log_inputs.exp()
        if text_to_image:
            gamma_sign = torch.sign(self.text_to_imagegamma)
            pt = gamma_sign * pt
            gamma_mag = abs(self.text_to_imagegamma).detach()
            weight_targets = torch.clamp(abs(labels - pt),min=1e-5,max=2)**gamma_mag
        else:
            gamma_sign = torch.sign(self.image_to_textgamma)
            pt = gamma_sign * pt
            gamma_mag = abs(self.image_to_textgamma).detach()
            weight_targets = torch.clamp(abs(labels - pt),min=1e-5,max=2)**gamma_mag
        loss = weight_targets * loss
        # loss = -torch.sum(F.log_softmax(inputs, dim=1)*weight_targets*labels,dim=1).mean()
        return loss.mean()