#  <center>Image Anything
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuromorph/image-anything/blob/main/image_anything.ipynb)  
A gradio demo of image models. Refer to [colab notebook](image_anything.ipynb) for setup.  

The project incorporates image generation and editing models in concert to perform various tasks.  


Currently Supported Tasks:
1. Image captioning
2. Generate auto SAM mask
3. Objects detection, segmentation, annotation
4. Remove / replace background
5. Inpainting
6. Upscale image 4x
7. Text to image 
8. Drawing to image
9. Image to image  
</br>

Interaction modes:
* Selecting points on the image
* Text prompts
* Auto mode
* Drawing
* Upload image mask -> TBD
* Audio -> TBD  
</br>

Models used:
* Segment Anything (SAM)
* Grounding DINO
* Matte Anything (ViTMatte - Hust Labs)
* Stable Diffusion 2 (Hugging Face diffusers)
* Stable Diffusion Controlnet 
* BLIP
* Mobile SAM
* Matte Anything Model (MAM - SHI Labs) -> TBD  
</br>

TBD: 
* Options to choose from checkpoints e.g. Stable Diffusion versions
* Options to further control SD generation
* More tasks e.g. image editing with more models
* 

</br>

###  App Snaps
Auto SAM Mask: 
![app screen](assets/first_screen.png) 

Upscale Task and Text to Image Task:
![txt2img upscale](assets/ups_t2i.png)  

Annotations: 
![anns](assets/anns.png)  

Inpainting (tea pot -> puppy ||  green apple -> orange || cat -> rabbit):  
![inpaint](assets/inp.png)  

Remove/Replace Background (SD generated backgrounds):  
![bgr](assets/bgn.png)  
Remove Background for Transparent objects:
![bg transp](assets/bgtr.png)  

Drawing to Image:  
![draw to img](assets/d2i.png)  

Image to Image A. (prompt for terrace swimming pool):
![img to img 1](assets/i2i1.png)  

Image to Image B. (prompt for 1: pool table with balls, 2:fantasy landscape on artstation):
![img to img 2](assets/i2i2.png)  

Advanced Settings to tune the results:  
![settings](assets/sett.png) 

## Acknowledgements
* [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) (inspiration and helpers)
* [Segment Anything](https://github.com/facebookresearch/segment-anything/)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* Stable Diffusion with [Hugging Face diffusers](https://github.com/huggingface/diffusers)
* [Controlnet](https://github.com/lllyasviel/ControlNet)
* [Matte-Anything](https://github.com/hustvl/Matte-Anything)
* [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM)
* [BLIP](https://arxiv.org/abs/2201.12086)