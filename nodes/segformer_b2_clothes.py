import os
import numpy as np
from urllib.request import urlopen
import torchvision.transforms as transforms  
import folder_paths
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image,ImageOps, ImageFilter
import torch.nn as nn
import torch

from .func import *

# 指定本地分割模型文件夹的路径
segformer_model_path=get_comfyui_config_model_path("segformer")

model_folder_path = os.path.join(segformer_model_path,"segformer_b2_clothes")




class segformer_b2_clothes:
   
    def __init__(self):
        pass
    
    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                 "image":("IMAGE",),
                "Face": ("BOOLEAN", {"default": True, "label_on": "✔ 脸部", "label_off": "× 脸部"}),
                "Hat": ("BOOLEAN", {"default": True, "label_on": "✔ 帽子", "label_off": "× 帽子"}),
                "Hair": ("BOOLEAN", {"default": True, "label_on": "✔ 头发", "label_off": "× 头发"}),
                "Upper_clothes": ("BOOLEAN", {"default": True, "label_on": "✔ 上衣", "label_off": "× 上衣"}),
                "Skirt": ("BOOLEAN", {"default": True, "label_on": "✔ 裙子", "label_off": "× 裙子"}),
                "Pants": ("BOOLEAN", {"default": True, "label_on": "✔ 裤子", "label_off": "× 裤子"}),
                "Dress": ("BOOLEAN", {"default": True, "label_on": "✔ 连衣裙", "label_off": "× 连衣裙"}),
                "Belt": ("BOOLEAN", {"default": True, "label_on": "✔ 皮带", "label_off": "× 皮带"}),
                "shoe": ("BOOLEAN", {"default": True, "label_on": "✔ 鞋子", "label_off": "× 鞋子"}),
                "leg": ("BOOLEAN", {"default": True, "label_on": "✔ 腿", "label_off": "× 腿"}),
                "arm": ("BOOLEAN", {"default": True, "label_on": "✔ 手臂", "label_off": "× 手臂"}),
                "Bag": ("BOOLEAN", {"default": True, "label_on": "✔ 包", "label_off": "× 包"}),
                "Scarf": ("BOOLEAN", {"default": True, "label_on": "✔ 围巾", "label_off": "× 围巾"})
                }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "♾️Mixlab/TryOn"

    def sample(self,image,Face,Hat,Hair,Upper_clothes,Skirt,Pants,Dress,Belt,shoe,leg,arm,Bag,Scarf):
        processor = SegformerImageProcessor.from_pretrained(model_folder_path)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)

        # 切割服装
        def get_segmentation(tensor_image):
            cloth = tensor2pil(tensor_image)
            # 预处理和预测
            inputs = processor(images=cloth, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
            return pred_seg,cloth

        results = []
        for item in image:
        
            # seg切割结果，衣服pil
            pred_seg,cloth = get_segmentation(item)
            labels_to_keep = [0]
            # if background :
            #     labels_to_keep.append(0)
            if not Hat:
                labels_to_keep.append(1)
            if not Hair:
                labels_to_keep.append(2)
            if not Upper_clothes:
                labels_to_keep.append(4)
            if not Skirt:
                labels_to_keep.append(5)
            if not Pants:
                labels_to_keep.append(6)
            if not Dress:
                labels_to_keep.append(7)
            if not Belt:
                labels_to_keep.append(8)
            if not shoe:
                labels_to_keep.append(9)
                labels_to_keep.append(10)
            if not Face:
                labels_to_keep.append(11)
            if not leg:
                labels_to_keep.append(12)
                labels_to_keep.append(13)
            if not arm:
                labels_to_keep.append(14) 
                labels_to_keep.append(15) 
            if not Bag:
                labels_to_keep.append(16)
            if not Scarf:
                labels_to_keep.append(17)
                
            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
            
            # 创建agnostic-mask图像
            mask_image = Image.fromarray(mask * 255)
            mask_image = mask_image.convert("RGB")
            mask_image = pil2tensor(mask_image)

            temp = (torch.clamp(mask_image, 0, 1.0) * 255.0).round().to(torch.int)
            temp = torch.bitwise_left_shift(temp[:,:,:,0], 16) + torch.bitwise_left_shift(temp[:,:,:,1], 8) + temp[:,:,:,2]
            mask = torch.where(temp == 0, 255, 0).float()

            results.append(mask)

        return (torch.cat(results, dim=0),)

NODE_CLASS_MAPPINGS = {
    "FashionClothMask2": segformer_b2_clothes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FashionClothMask2": "Fashion Cloth Mask 2"
}