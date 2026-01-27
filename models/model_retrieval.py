from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .open_clip.factory import create_model_and_transforms
from .open_clip.transformer import text_global_pool, VisionTransformer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import gc
from .open_clip.transformer import MultimodalTransformer
import pdb
from transformers import BertConfig
from models.open_clip import tokenizer
from .xbert import BertModel

def create_and_load_pretrained(config=None):
    model_params = {
        "model_name": config.get("model_name", "ViT-B/32") if config else "ViT-B/32",
    }
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_params["model_name"],
        pretrained=None
    )
    return model,preprocess_train, preprocess_val


class CLIPFusionModule(nn.Module):
    def __init__(
        self, 
        config=None,
        image_dim: int = 768,
        text_dim: int = 512,
        fusion_dim: int = 512, 
        alpha: float = 0.1,
        temperature=0.07, 
        margin=0.2
    ):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.alpha = alpha
        self.clip, self.preprocess_train, self.preprocess_val = create_and_load_pretrained(config)


    def encode_image(self, image, embeds=False):
        return self.clip.encode_image(image, embeds=embeds)
    
    def encode_text(self, text, embeds=False):
        return self.clip.encode_text(text, embeds=embeds)
    
    def encode_weight_image(self, text, image_embed):
        itm_input = self.clip.encode_weight_image(text, image_embed)
        score = self.clip.itm_head(itm_input)[:, 1]
        return score
    
    def get_logits(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_logits =  self.clip.logit_scale.exp() * image_features @ text_features.T
        if self.clip.logit_bias is not None:
            image_logits += self.clip.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits
    
    def get_text_to_image_mapping(self, text_features, image_features, k):
        similarity = text_features @ image_features.T
        sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
        topk_indices = sorted_indices[:,:k]
        return topk_indices
    
    def get_image_to_text_mapping(self, image_features, text_features, k):
        similarity = image_features @ text_features.T
        sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
        topk_indices = sorted_indices[:,:k]
        return topk_indices
    
    def forward(self, image, text,WeightsoftCEloss):
        total_loss = self.clip(image, text,WeightsoftCEloss)['total_loss']
        return total_loss