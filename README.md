#  <center>Image Anything
A gradio demo of image models. Refer to [colab notebook](image_anything.ipynb) for setup.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuromorph/image-anything/blob/main/image-anything.ipynb)

The project incorporates image generation and editing models in concert to perform various tasks.  


Currently Supported Tasks:  
1. Image captioning
2. Generate auto SAM mask
3. Objects detection, segmentation, annotation
4. Remove / replace background
5. Inpainting
6. Upscale image 4x
7. Text to image  
</br>

Interaction modes:
* Selecting points on the image
* Text prompts
* Auto mode
* Input image mask -> TBD
* Audio -> TBD  
</br>

Models used:
* Segment Anything (SAM)
* Grounding DINO
* Matte Anything (ViTMatte - Hust Labs)
* Stable Diffusion 2 (Hugging Face diffusers)
* BLIP
* Mobile SAM
* Matte Anything Model (MAM - SHI Labs) -> TBD  
</br>

TBD: 
* Options to choose from checkpoints e.g. Stable Diffusion versions
* Options to control SD generation
* More tasks e.g. image editing with more models
* 

</br>

![txt2img](assets/text2img.png)
![upscale](assets/upscale.png)
![ann1](assets/ann1.png)
![bg1](assets/bg1.png)
![inpaint](assets/inpaint.png)
![transp1](assets/transp1.png)
![transp2](assets/transp2.png)
![set1](assets/settings1.png) ![set2](assets/settings2.png)

## Acknowledgements
* [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) (inspiration and helpers)
* [Segment Anything](https://github.com/facebookresearch/segment-anything/)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* Stable Diffusion with [Hugging Face diffusers](https://github.com/huggingface/diffusers)
* [Matte-Anything](https://github.com/hustvl/Matte-Anything)
* [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM)
* [BLIP](https://arxiv.org/abs/2201.12086)