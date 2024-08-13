# ComfyUI Try-On: Virtual Try-On for Creating a Personal Brand Wardrobe Collection

ComfyUI虚拟试穿：为创建个人品牌衣橱集合的虚拟试穿系统

This project provides a virtual try-on system using ComfyUI to help create a personal brand wardrobe collection. Below are the details on how to set up and use the models required for the project.

本项目提供了一个使用ComfyUI的虚拟试穿系统，帮助创建个人品牌衣橱集合。以下是设置和使用所需模型的详细信息。

## Nodes
节点

- CatVTONNode
- FashionClothMask
- FashionClothMask2

## Models
模型

### CatVTON Model
CatVTON模型

1. Download the CatVTON model from the provided Baidu Netdisk link:
   从提供的百度网盘链接下载CatVTON模型：
   - **Link 链接**: [CatVTON Model](https://pan.baidu.com/s/1rajZvsgEBE9seNFjq935iQ?pwd=MAI0)
   - **Extraction Code 提取码**: MAI0

2. Place the downloaded model in the following directory:
   将下载的模型放在以下目录：
```

ComfyUI/models/catvton

```

### Segformer Models
Segformer模型

1. **Segformer B3 Fashion**:
- Download the model from Hugging Face:
  从Hugging Face下载模型：
  - **Link 链接**: [Segformer B3 Fashion](https://huggingface.co/sayeed99/segformer-b3-fashion)
- Place the downloaded model in the following directory:
  将下载的模型放在以下目录：
  ```
  ComfyUI/models/segformer/segformer-b3-fashion
  ```

2. **Segformer B2 Clothes**:

- Download the model from Hugging Face:
  从Hugging Face下载模型：
  - **Link 链接**: [Segformer B2 Clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)

- Place the downloaded model in the following directory:
  将下载的模型放在以下目录：
  ```
  ComfyUI/models/segformer/segformer_b2_clothes
  ```

## References
参考

- **CatVTON**: [CatVTON](https://github.com/Zheng-Chong/CatVTON)
- **ComfyUI Workflow**: [ComfyUI Workflow](https://github.com/Zheng-Chong/CatVTON?tab=readme-ov-file#comfyui-workflow)
- **ComfyUI CatVTON Wrapper**: [ComfyUI CatVTON Wrapper](https://github.com/chflame163/ComfyUI_CatVTON_Wrapper)
- **Segformer B2 Clothes**: [Segformer B2 Clothes](https://github.com/StartHua/Comfyui_segformer_b2_clothes)

