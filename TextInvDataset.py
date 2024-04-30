<<<<<<< HEAD
import torch
import numpy as np
import PIL
import transformers
from torch.utils.data import Dataset
from torch import nn
from torchvision import transforms
from PIL import Image

import os
import random

class TextualInversionDataset(Dataset):
    """Dataset for style/content training, implemented by Shangjun Meng and Jirong Yang"""

    def __init__(self,
                 dir_path,
                 attribute,
                 tokenizer,
                 placeholder_token='<>',
                 size=768,
                 operation='train',
                 interpolation='BL',
                 repeat=50,
                 flip=True,
                 gray=False,
                 prompt_list=None,
                 ) -> None:
        """
        dir_path: directory of images subject to current operation
        attribute: "content" or "style"
        placeholder_token: Anything that is not already in pretrained CLIP token dict is fine, wrap with <> just in case
        size: standardized input size--size*size
        operation: "train" or "transfer"
        interpolation: how to resize, "NN" "BL" and "BC" to choose from (nearest neighbor, bilinear, bicubic, fastest -> slowest)
        repeat: #times we train on the same image
        flip: horizontally flip image to expand train size
        gray: make grayscales of original image to expand train size, use when only shape matters.
        prompt_list: list of text prompts containing inverted token we want model to correlate the attribute-only picture as, 
                    if none then condition solely with inverted token
        """
        super().__init__()

        self.image_path_list = [os.path.join(dir_path, img_name) for img_name in os.listdir(dir_path)]
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.attr = attribute
        self.size = size
        self.prompt_list = prompt_list

        self._length = len(self.image_path_list)

        if operation=="train":
            self._length = self._length * repeat

        self.interpolation = {'NN': Image.NEAREST, 'BL': Image.BILINEAR, 'BC': Image.BICUBIC}[interpolation]
        self.flip = flip
        self.flip_transform = transforms.Compose([transforms.RandomHorizontalFlip()])

        
    #basic methods
    def __len__(self):
        return self._length
    
    def __str__(self):
        return f"TextInvDataset(attr: {self.attr}, input size: {self.size}*{self.size}, set size: {len(self)})"
    

    def __getitem__(self, idx):
        i = idx % len(self.image_path_list) #no over index
        img = Image.open(self.image_path_list[i])

        placeholder = self.placeholder_token

        if self.prompt_list:
            cond_text = random.choice(self.prompt_list).format(placeholder)
        else:
            cond_text = placeholder

        instance = {}

        ##tokenization of placeholder/prompt##
        instance['input_ids'] = self.tokenizer(
            cond_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt").input_ids[0]
        

        img = img.resize((self.size, self.size), resample=self.interpolation)
        img = self.flip_transform(img)

        #normalize to (0,1)#
        image = np.array(img).astype(np.uint8)
        image = (image / (255/2.0) - 1.0).astype(np.float32)

        #np(h,w,c) -> torch(c,h,w)
        instance['pix_values'] = torch.from_numpy(image).permute(2,0,1)

        return instance

=======
import torch
import numpy as np
import PIL
import transformers
from torch.utils.data import Dataset
from torch import nn
from torchvision import transforms
from PIL import Image

import os
import random

class TextualInversionDataset(Dataset):
    """Dataset for style/content training, implemented by Shangjun Meng and Jirong Yang"""

    def __init__(self,
                 dir_path,
                 attribute,
                 tokenizer,
                 placeholder_token='<>',
                 size=768,
                 operation='train',
                 interpolation='BL',
                 repeat=50,
                 flip=True,
                 gray=False,
                 prompt_list=None,
                 ) -> None:
        """
        dir_path: directory of images subject to current operation
        attribute: "content" or "style"
        placeholder_token: Anything that is not already in pretrained CLIP token dict is fine, wrap with <> just in case
        size: standardized input size--size*size
        operation: "train" or "transfer"
        interpolation: how to resize, "NN" "BL" and "BC" to choose from (nearest neighbor, bilinear, bicubic, fastest -> slowest)
        repeat: #times we train on the same image
        flip: horizontally flip image to expand train size
        gray: make grayscales of original image to expand train size, use when only shape matters.
        prompt_list: list of text prompts containing inverted token we want model to correlate the attribute-only picture as, 
                    if none then condition solely with inverted token
        """
        super().__init__()

        self.image_path_list = [os.path.join(dir_path, img_name) for img_name in os.listdir(dir_path)]
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.attr = attribute
        self.size = size
        self.prompt_list = prompt_list

        self._length = len(self.image_path_list)

        if operation=="train":
            self._length = self._length * repeat

        self.interpolation = {'NN': Image.NEAREST, 'BL': Image.BILINEAR, 'BC': Image.BICUBIC}[interpolation]
        self.flip = flip
        self.flip_transform = transforms.Compose([transforms.RandomHorizontalFlip()])

        
    #basic methods
    def __len__(self):
        return self._length
    
    def __str__(self):
        return f"TextInvDataset(attr: {self.attr}, input size: {self.size}*{self.size}, set size: {len(self)})"
    

    def __getitem__(self, idx):
        i = idx % len(self.image_path_list) #no over index
        img = Image.open(self.image_path_list[i])

        placeholder = self.placeholder_token

        if self.prompt_list:
            cond_text = random.choice(self.prompt_list).format(placeholder)
        else:
            cond_text = placeholder

        instance = {}

        ##tokenization of placeholder/prompt##
        instance['input_ids'] = self.tokenizer(
            cond_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt").input_ids[0]
        

        img = img.resize((self.size, self.size), resample=self.interpolation)
        img = self.flip_transform(img)

        #normalize to (0,1)#
        image = np.array(img).astype(np.uint8)
        image = (image / (255/2.0) - 1.0).astype(np.float32)

        #np(h,w,c) -> torch(c,h,w)
        instance['pix_values'] = torch.from_numpy(image).permute(2,0,1)

        return instance

>>>>>>> d768bb3 (relocated from GDrive, UNet safetensors removed for upload)
