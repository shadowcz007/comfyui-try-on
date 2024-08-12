# 修改自https://github.com/StartHua/Comfyui_segformer_b2_clothes/blob/main/segformer_b3_fashion.py

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

model_folder_path = os.path.join(segformer_model_path,"segformer-b3-fashion")



class segformer_b3_fashion:
   
    def __init__(self):
        pass
    
    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                "image":("IMAGE",), 
               "shirt": ("BOOLEAN", {"default": False, "label_on": "✔ 衬衫、罩衫", "label_off": "× 衬衫、罩衫"}),
                "top": ("BOOLEAN", {"default": False, "label_on": "✔ 上衣、t恤", "label_off": "× 上衣、t恤"}),
                "sweater": ("BOOLEAN", {"default": False, "label_on": "✔ 毛衣", "label_off": "× 毛衣"}),
                "cardigan": ("BOOLEAN", {"default": False, "label_on": "✔ 开襟羊毛衫", "label_off": "× 开襟羊毛衫"}),
                "jacket": ("BOOLEAN", {"default": False, "label_on": "✔ 夹克", "label_off": "× 夹克"}),
                "vest": ("BOOLEAN", {"default": False, "label_on": "✔ 背心", "label_off": "× 背心"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "✔ 裤子", "label_off": "× 裤子"}),
                "shorts": ("BOOLEAN", {"default": False, "label_on": "✔ 短裤", "label_off": "× 短裤"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "✔ 裙子", "label_off": "× 裙子"}),
                "coat": ("BOOLEAN", {"default": False, "label_on": "✔ 外套", "label_off": "× 外套"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "✔ 连衣裙", "label_off": "× 连衣裙"}),
                "jumpsuit": ("BOOLEAN", {"default": False, "label_on": "✔ 连身裤", "label_off": "× 连身裤"}),
                "cape": ("BOOLEAN", {"default": False, "label_on": "✔ 斗篷", "label_off": "× 斗篷"}),
                "glasses": ("BOOLEAN", {"default": False, "label_on": "✔ 眼镜", "label_off": "× 眼镜"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "✔ 帽子", "label_off": "× 帽子"}),
                "hairaccessory": ("BOOLEAN", {"default": False, "label_on": "✔ 头带", "label_off": "× 头带"}),
                "tie": ("BOOLEAN", {"default": False, "label_on": "✔ 领带", "label_off": "× 领带"}),
                "glove": ("BOOLEAN", {"default": False, "label_on": "✔ 手套", "label_off": "× 手套"}),
                "watch": ("BOOLEAN", {"default": False, "label_on": "✔ 手表", "label_off": "× 手表"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "✔ 皮带", "label_off": "× 皮带"}),
                "legwarmer": ("BOOLEAN", {"default": False, "label_on": "✔ 暖腿器", "label_off": "× 暖腿器"}),
                "tights": ("BOOLEAN", {"default": False, "label_on": "✔ 紧身衣、长筒袜", "label_off": "× 紧身衣、长筒袜"}),
                "sock": ("BOOLEAN", {"default": False, "label_on": "✔ 袜子", "label_off": "× 袜子"}),
                "shoe": ("BOOLEAN", {"default": False, "label_on": "✔ 鞋子", "label_off": "× 鞋子"}),
                "bagwallet": ("BOOLEAN", {"default": False, "label_on": "✔ 包、钱包", "label_off": "× 包、钱包"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "✔ 围巾", "label_off": "× 围巾"}),
                "umbrella": ("BOOLEAN", {"default": False, "label_on": "✔ 雨伞", "label_off": "× 雨伞"}),
                "hood": ("BOOLEAN", {"default": False, "label_on": "✔ 兜帽", "label_off": "× 兜帽"}),
                "collar": ("BOOLEAN", {"default": False, "label_on": "✔ 衣领", "label_off": "× 衣领"}),
                "lapel": ("BOOLEAN", {"default": False, "label_on": "✔ 翻领", "label_off": "× 翻领"}),
                "epaulette": ("BOOLEAN", {"default": False, "label_on": "✔ 肩章", "label_off": "× 肩章"}),
                "sleeve": ("BOOLEAN", {"default": False, "label_on": "✔ 袖子", "label_off": "× 袖子"}),
                "pocket": ("BOOLEAN", {"default": False, "label_on": "✔ 口袋", "label_off": "× 口袋"}),
                "neckline": ("BOOLEAN", {"default": False, "label_on": "✔ 领口", "label_off": "× 领口"}),
                "buckle": ("BOOLEAN", {"default": False, "label_on": "✔ 带扣", "label_off": "× 带扣"}),
                "zipper": ("BOOLEAN", {"default": False, "label_on": "✔ 拉链", "label_off": "× 拉链"}),
                "applique": ("BOOLEAN", {"default": False, "label_on": "✔ 贴花", "label_off": "× 贴花"}),
                "bead": ("BOOLEAN", {"default": False, "label_on": "✔ 珠子", "label_off": "× 珠子"}),
                "bow": ("BOOLEAN", {"default": False, "label_on": "✔ 蝴蝶结", "label_off": "× 蝴蝶结"}),
                "flower": ("BOOLEAN", {"default": False, "label_on": "✔ 花", "label_off": "× 花"}),
                "fringe": ("BOOLEAN", {"default": False, "label_on": "✔ 额前短垂发", "label_off": "× 额前短垂发"}),
                "ribbon": ("BOOLEAN", {"default": False, "label_on": "✔ 丝带", "label_off": "× 丝带"}),
                "rivet": ("BOOLEAN", {"default": False, "label_on": "✔ 铆钉", "label_off": "× 铆钉"}),
                "ruffle": ("BOOLEAN", {"default": False, "label_on": "✔ 褶饰", "label_off": "× 褶饰"}),
                "sequin": ("BOOLEAN", {"default": False, "label_on": "✔ 亮片", "label_off": "× 亮片"}),
                "tassel": ("BOOLEAN", {"default": False, "label_on": "✔ 流苏", "label_off": "× 流苏"})


                }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "♾️Mixlab/TryOn"

    def sample(self,image,
        shirt,
        top,
        sweater,
        cardigan,
        jacket,
        vest,
        pants,
        shorts,
        skirt,
        coat,
        dress,
        jumpsuit,
        cape,
        glasses,
        hat,
        hairaccessory,
        tie,
        glove,
        watch,
        belt,
        legwarmer,
        tights,
        sock,
        shoe,
        bagwallet,
        scarf,
        umbrella,
        hood,
        collar,
        lapel,
        epaulette,
        sleeve,
        pocket,
        neckline,
        buckle,
        zipper,
        applique,
        bead,
        bow,
        flower,
        fringe,
        ribbon,
        rivet,
        ruffle,
        sequin,
        tassel):

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
            if not shirt:
                labels_to_keep.append(1)
            if not top:
                labels_to_keep.append(2)
            if not sweater:
                labels_to_keep.append(3)
            if not cardigan:
                labels_to_keep.append(4)
            if not jacket:
                labels_to_keep.append(5)
            if not vest:
                labels_to_keep.append(6)
            if not pants:
                labels_to_keep.append(7)
            if not shorts:
                labels_to_keep.append(8)
            if not skirt:
                labels_to_keep.append(9)
            if not coat:
                labels_to_keep.append(10)
            if not dress:
                labels_to_keep.append(11)
            if not jumpsuit:
                labels_to_keep.append(12)
            if not cape:
                labels_to_keep.append(13)
            if not glasses:
                labels_to_keep.append(14)
            if not hat:
                labels_to_keep.append(15)
            if not hairaccessory:
                labels_to_keep.append(16)
            if not tie:
                labels_to_keep.append(17)
            if not glove:
                labels_to_keep.append(18)
            if not watch:
                labels_to_keep.append(19)
            if not belt:
                labels_to_keep.append(20)
            if not legwarmer:
                labels_to_keep.append(21)
            if not tights:
                labels_to_keep.append(22)
            if not sock:
                labels_to_keep.append(23)
            if not shoe:
                labels_to_keep.append(24)
            if not bagwallet:
                labels_to_keep.append(25)
            if not scarf:
                labels_to_keep.append(26)
            if not umbrella:
                labels_to_keep.append(27)
            if not hood:
                labels_to_keep.append(28)
            if not collar:
                labels_to_keep.append(29)
            if not lapel:
                labels_to_keep.append(30)
            if not epaulette:
                labels_to_keep.append(31)
            if not sleeve:
                labels_to_keep.append(32)
            if not pocket:
                labels_to_keep.append(33)
            if not neckline:
                labels_to_keep.append(34)
            if not buckle:
                labels_to_keep.append(35)
            if not zipper:
                labels_to_keep.append(36)
            if not applique:
                labels_to_keep.append(37)
            if not bead:
                labels_to_keep.append(38)
            if not bow:
                labels_to_keep.append(39)
            if not flower:
                labels_to_keep.append(40)
            if not fringe:
                labels_to_keep.append(41)
            if not ribbon:
                labels_to_keep.append(42)
            if not rivet:
                labels_to_keep.append(43)
            if not ruffle:
                labels_to_keep.append(44)
            if not sequin:
                labels_to_keep.append(45)
            if not tassel:
                labels_to_keep.append(46)
                
            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
            
            # 创建agnostic-mask图像
            mask_image = Image.fromarray(mask * 255)
            mask_image = mask_image.convert("RGB")

            mask_image = pil2tensor(mask_image)

            temp = (torch.clamp(mask_image, 0, 1.0) * 255.0).round().to(torch.int)
            temp = torch.bitwise_left_shift(temp[:,:,:,0], 16) + torch.bitwise_left_shift(temp[:,:,:,1], 8) + temp[:,:,:,2]
            mask = torch.where(temp == 0, 255, 0).float()

            results.append(mask)

        
        processor=None
        model=None

        return (torch.cat(results, dim=0),)

NODE_CLASS_MAPPINGS = {
    "FashionClothMask": segformer_b3_fashion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FashionClothMask": "Fashion Cloth Mask"
}