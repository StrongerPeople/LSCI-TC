import json
import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from dataset.utils import pre_caption

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=77):
        self.ann = []
        self.text       = []
        self.image      = []
        self.txt2img    = {}
        self.img2txt    = {}
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['filename'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['captions']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['filename'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.text[np.random.choice(self.img2txt[index])]
        return image, caption


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=256):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        # self.mask_text = []
        self.image = []
        # self.image_data = []
        self.txt2img = {}
        self.img2txt = {}
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['filename'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['captions']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

                # t = analyse.extract_tags(caption, topK=4, withWeight=False)
                # ii = caption.split(' ')
                # k = ""
                # fl = 0
                # for j in range(len(ii)):
                #     if fl == 1:
                #         k += " "
                #     fl = 1
                #     if ii[j] not in t:
                #         k += "[MASK]"
                #     else:
                #         k += ii[j]
                # self.mask_text.append(pre_caption(k, self.max_words))

                # image_path = os.path.join(self.image_root, ann['image'])
                # image = Image.open(image_path).convert('RGB')
                # image = self.transform(image)
                # self.image_data.append(image.unsqueeze(dim=0))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.text[random.choice(self.img2txt[index])]

        return image, caption