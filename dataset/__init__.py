import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.re_dataset import re_train_dataset, re_eval_dataset
from dataset.pretrain_dataset import ImageTextJsonDataset, RegionTextJsonDataset
from models.open_clip.tokenizer import tokenize
from dataset.randaugment import RandomAugment
from torchvision.transforms import InterpolationMode
from dataset.grounding_dataset import grounding_dataset


def create_dataset(dataset, config, evaluate=False, train_transformer = None, val_transformer = None):

    train_transform = train_transformer
    test_transform = val_transformer

    train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
    val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
    test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
    return train_dataset, val_dataset, test_dataset



def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
            
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last
        )
        
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders

def dataset_collate(batch):
    images      = []
    captions    = []
    for image, caption in batch:
        images.append(image)
        captions.append(caption)
    captions = tokenize(captions)
    images   = torch.stack(images,dim=0)
    
    return images, captions

def rs5m_dataset_collate(batch):
    images      = []
    captions    = []
    for (image,caption) in batch:
        images.append(image)
        captions.append(caption)
    captions = tokenize(captions)
    images   = torch.stack(images,dim=0)
    
    return images, captions
