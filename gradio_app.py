import os
import random
import sys

import gradio as gr
import argparse

import numpy as np
import torch
import torchvision
from torchvision.ops import box_convert
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import cv2

# Grounding DINO
sys.path.insert(0, './GroundingDINO')
from groundingdino.util.inference import Model

# segment anything
from segment_anything import sam_model_registry as smr, SamPredictor as sp, SamAutomaticMaskGenerator as smg
from mobile_sam import sam_model_registry as smr_mb, SamPredictor as sp_mb, SamAutomaticMaskGenerator as smg_mb
import numpy as np

# diffusers
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

# Matte Anything (VitMatte)
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

# Controlnet Drawing Scribble
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swint_ogc.pth"

SD_GEN_CHECKPOINT = "stabilityai/stable-diffusion-2-1-base"
SD_INP_CHECKPOINT = "stabilityai/stable-diffusion-2-inpainting"
SD_UPS_CHECKPOINT = "stabilityai/stable-diffusion-x4-upscaler"
SD_I2I_CHECKPOINT = "stabilityai/stable-diffusion-2-1-base"
SD_CNTRLNET_CKP = "runwayml/stable-diffusion-v1-5"
CONTROLNET_SCRIB_CKP = "lllyasviel/control_v11p_sd15_scribble"

sam_import = {
	'SAM - Meta': {'sam_model_registry': smr, 'SamPredictor': sp, 'SamAutomaticMaskGenerator': smg},
    'Mobile SAM': {'sam_model_registry': smr_mb, 'SamPredictor': sp_mb, 'SamAutomaticMaskGenerator': smg_mb} 
}
sam_config = {
	'SAM - Meta': 'vit_h', 
    'Mobile SAM': 'vit_t'  
}
sam_models = {
	'vit_h': './checkpoints/sam_vit_h_4b8939.pth', # Meta SAM
    'vit_t': './checkpoints/mobile_sam.pt'  # Mobile SAM
}

matte_config = {
	'vit_b': 'Matte_Anything/configs/matte_anything.py'
}
matte_models = {
	'vit_b': './checkpoints/ViTMatte_B_DIS.pth'
}

GR_PALETTE = (51, 255, 146)

blip_processor = None
blip_model = None
groundingdino_model = None
sam_predictor = None
sam_automask_generator = None
sd_inp_pipeline = None
sd_gen_pipeline = None
sd_ups_pipeline = None
sd_cn_pipeline = None
sd_i2i_pipeline = None
vitmatte = None
caption = None
attn_slicing = True # enable_attention_slicing
dpm_scheduler = True # use DPMSolver Multistep Scheduler
output_dir="outputs"
device="cuda"
sd_kwargs = {
    'torch_dtype':torch.float16,
    # 'safety_checker':None,
    # 'feature_extractor':None,
    # 'requires_safety_checker':False
}


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img*255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    return full_img, res

def generate_caption(processor, blip_model, raw_image):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=2)

    if label:
        font = ImageFont.load_default()
        # font = ImageFont.truetype("assets/OpenSans-Regular.ttf", 50)
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")

        draw.text((box[0], box[1]), label)

def init_vitmatte(model_type):
    #Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
    cfg = LazyConfig.load(matte_config[model_type])
    vitmatte = instantiate(cfg.model)
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(matte_models[model_type])
    return vitmatte

def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated==255] = 128
    trimap[eroded==255] = 255
    return trimap

def convert_pixels(gray_image, boxes):
    converted_image = np.copy(gray_image)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        converted_image[y1:y2, x1:x2][converted_image[y1:y2, x1:x2] == 1] = 0.5
    return converted_image

def clear_old():
    global caption
    caption = None # Set caption to None on new image upload
    pass

def create_canvas():
    return np.zeros(shape=(1000, 1000, 3), dtype=np.uint8) + 255


def run_image_anything(input_image, task_type, text_prompt, sd_prompt, negative_prompt, scribble_mode, box_threshold, text_threshold, iou_threshold, 
                                                      erode_kernel_size, dilate_kernel_size, tr_prompt, tr_box_threshold, tr_text_threshold, inpaint_mode, 
                                                      model_matte, model_sam, guidance_scale, strength, num_inference_steps):
    
    global blip_processor, blip_model, groundingdino_model, sam_predictor, sam_automask_generator, sd_inp_pipeline, sd_gen_pipeline, \
    sd_ups_pipeline, sd_cn_pipeline, sd_i2i_pipeline, sd_kwargs, attn_slicing, dpm_scheduler, vitmatte, caption, device

    torch.cuda.empty_cache()

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    if task_type != "Text-to-Image":
        # load image
        image = input_image["image"]
        scribble = input_image["mask"]
        size = image.size # w, h
        image_pil = image.convert("RGB")
        image = np.array(image_pil)
    
    # Initialize SAM
    if sam_predictor is None:
        model_type = sam_config[model_sam]
        assert sam_models[model_type], 'SAM checkpoint is not found for the selected SAM model!'
        sam = sam_import[model_sam]['sam_model_registry'][model_type](checkpoint=sam_models[model_type])
        sam.to(device)
        sam.eval()
        sam_predictor = sam_import[model_sam]['SamPredictor'](sam)
        sam_automask_generator = sam_import[model_sam]['SamAutomaticMaskGenerator'](sam)

    # Initialize GroundingDINO
    if groundingdino_model is None:
        groundingdino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=device)


    if task_type == "Image Caption":
        if caption is None:
            # generate caption 
            blip_processor = blip_processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            blip_model = blip_model or BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
            caption = generate_caption(blip_processor, blip_model, image_pil)
        return [caption, []]

    if task_type == "Auto SAM Mask":
        masks = sam_automask_generator.generate(image)
        full_img, res = show_anns(masks)
        return [caption, [ (full_img, "Auto SAM Mask")]]

    if task_type == "Detection/Annotation/Segmentation" or task_type == "Inpainting" or task_type == "Remove/Replace Background":
        sam_predictor.set_image(image)
        point_coords, point_labels, transformed_boxes = None, None, None

        scribble = scribble.convert("RGB")
        scribble = np.array(scribble)
        scribble = scribble.transpose(2, 1, 0)[0]
        # User selected regions (scribble)
        labeled_array, num_features = ndimage.label(scribble >= 255)
        

        if num_features > 0:          
            # Calculate centroid of regions
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)

            point_coords = torch.from_numpy(centers)
            point_coords = sam_predictor.transform.apply_coords_torch(point_coords, image.shape[:2])
            point_coords = point_coords.unsqueeze(0).to(device)
            point_labels = torch.from_numpy(np.array([1] * len(centers))).unsqueeze(0).to(device)
            if scribble_mode == 'split':
                point_coords = point_coords.permute(1, 0, 2)
                point_labels = point_labels.permute(1, 0)

        # get text prompt for grounding dino
        if text_prompt == "" or text_prompt is None:
            if caption == "" or caption is None:
                blip_processor = blip_processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                blip_model = blip_model or BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
                caption = generate_caption(blip_processor, blip_model, image_pil)

            text_prompt = caption
        
        # run grounding dino model
        detections, pred_phrases = groundingdino_model.predict_with_caption(
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
            caption = text_prompt, 
            box_threshold = box_threshold, 
            text_threshold = text_threshold
        )

        # use NMS to handle overlapped boxes
        if len(detections.xyxy) > 1:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                iou_threshold,
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            pred_phrases = np.array(pred_phrases)
            pred_phrases = pred_phrases[nms_idx]
        
        if num_features < 1:
            transformed_boxes = sam_predictor.transform.apply_boxes(detections.xyxy, image.shape[:2])
            transformed_boxes = torch.as_tensor(transformed_boxes, dtype=torch.float).to(device)

        # SAM masks
        masks, _, _ = sam_predictor.predict_torch(
                point_coords = point_coords,
                point_labels = point_labels,
                boxes = transformed_boxes,
                multimask_output = False,
            )

        if task_type == "Detection/Annotation/Segmentation":
            image_draw = ImageDraw.Draw(image_pil)
            boxdraw = True
            for box, label in zip(detections.xyxy, pred_phrases):               
                if num_features > 0:
                    boxdraw = False
                    # if any scribble point is inside box, draw that box
                    for c in centers:
                        if  c[0] >= box[0] and c[0] <= box[2] and c[1] >= box[1] and c[1] <= box[3]:
                            boxdraw = True
                            break
                if boxdraw:
                    draw_box(box, image_draw, label)

            mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)
            for mask in masks:
                draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

            image_pil = image_pil.convert('RGBA')
            image_pil.alpha_composite(mask_image)

            return [caption, [(image_pil, "Annotations/Segmentation"), (mask_image, "SAM Mask")]]        
            

        if task_type == "Inpainting":
            # assert sd_prompt, "Inpaint_prompt is required!"
            # assert num_features > 0 or text_prompt, "Text prompt or Selected/scribble points required!"

            if inpaint_mode == 'merge':
                masks_in = torch.sum(masks, dim=0)
                masks_in = torch.where(masks_in > 0, True, False)
                mask = masks_in[0].cpu().numpy()
                # struct2 = ndimage.generate_binary_structure(2, 2)
                # mask_dilated = ndimage.binary_dilation(mask, structure=struct2, iterations=3).astype(mask.dtype)
                masks_pil = [Image.fromarray(mask)]
            else: # inpaint_mode => 'split'
                masks_in = torch.where(masks > 0, True, False)
                masks_in = masks_in.cpu().numpy() 
                # struct2 = ndimage.generate_binary_structure(2, 2)
                # masks_dilated = [ndimage.binary_dilation(masks_in[i][0], structure=struct2, iterations=3).astype(masks_in[i][0].dtype) for i in range(masks_in.shape[0])]
                # masks_pil = [Image.fromarray(masks_dilated[i]) for i in range(len(masks_dilated))]
                masks_pil = [Image.fromarray(masks_in[i][0]) for i in range(masks_in.shape[0])]

            # inpainting pipeline
            if sd_inp_pipeline is None:
                sd_inp_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                SD_INP_CHECKPOINT, **sd_kwargs
                )
                sd_inp_pipeline = sd_inp_pipeline.to(device)
                if dpm_scheduler: sd_inp_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sd_inp_pipeline.scheduler.config)
                if attn_slicing: sd_inp_pipeline.enable_attention_slicing()
            
            inp_img = []
            for i in range(len(masks_pil)):
                img = sd_inp_pipeline(prompt=sd_prompt, negative_prompt=negative_prompt, image=image_pil.resize((512, 512)), mask_image=masks_pil[i].resize((512, 512)), num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength).images[0]
                img = img.resize(size)
                inp_img.append(img)

            return [caption, [(inp_img[i], "Inpainting") for i in range(len(inp_img))] + [(masks_pil[i], "SAM Mask") for i in range(len(masks_pil))]]


        if task_type == "Remove/Replace Background":

            if vitmatte is None:
                vitmatte = init_vitmatte('vit_b')
            masks = masks.cpu().detach().numpy()

            # generate alpha matte
            torch.cuda.empty_cache()
            mask = np.zeros(masks[0][0].shape)
            print("Mask shape ", mask.shape, masks.shape[0])
            for i in range(masks.shape[0]):
                mask[masks[i][0] == True] = 255
            mask = mask.astype(np.uint8)
            # print("Trimap in values: ", np.unique(mask))
            trimap = generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
            print("Trimap out values: ", np.unique(trimap))
            trimap[trimap==128] = 0.5
            trimap[trimap==255] = 1

            # run grounding dino model for transparency
            boxes, phrases = groundingdino_model.predict_with_caption(
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                caption = tr_prompt, 
                box_threshold = tr_box_threshold, 
                text_threshold = tr_text_threshold
            )

            if boxes.xyxy.shape[0] == 0:
                # no transparent object detected
                pass
            else:
                trimap = convert_pixels(trimap, boxes.xyxy)

            input = {
                "image": torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)/255,
                "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
            }

            torch.cuda.empty_cache()
            alpha = vitmatte(input)['phas'].flatten(0,2)
            alpha = alpha.detach().cpu().numpy()

            # get a green background
            background = np.array([GR_PALETTE], dtype='uint8')

            # calculate foreground with alpha blending
            foreground_alpha = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255
            foreground_alpha[foreground_alpha>1] = 1

            # return img, mask_all
            trimap[trimap==1] == 0.999

            # new background

            background_1 = cv2.imread('assets/sea.jpeg')
            background_1 = cv2.resize(background_1, (image.shape[1], image.shape[0]))

            # to RGB
            background_1 = cv2.cvtColor(background_1, cv2.COLOR_BGR2RGB)

            # use alpha blending
            new_bg_1 = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_1 * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

            if (sd_prompt is not None and sd_prompt != ""):
                if sd_gen_pipeline is None:
                    if sd_i2i_pipeline is not None:
                        sd_gen_pipeline = StableDiffusionPipeline.from_pretrained(**sd_i2i_pipeline.components)
                    else:
                        sd_gen_pipeline = StableDiffusionPipeline.from_pretrained(SD_GEN_CHECKPOINT, **sd_kwargs)
                
                    sd_gen_pipeline = sd_gen_pipeline.to(device)
                    if dpm_scheduler: sd_gen_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sd_gen_pipeline.scheduler.config)
                    if attn_slicing: sd_gen_pipeline.enable_attention_slicing()

                background_img = sd_gen_pipeline(prompt = sd_prompt, negative_prompt = negative_prompt,  guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
                background_img = np.array(background_img)
                background_img = cv2.resize(background_img, (image.shape[1], image.shape[0]))
                new_bg_sd = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_img * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255
                return [caption, [(mask, "SAM Mask"), (alpha, "Alpha Mask"), (foreground_alpha, "Foreground Alpha"), (new_bg_1, "New Sample Background"), (new_bg_sd, "New Diffusion Background")]]
            else:
                return [caption, [(mask, "SAM Mask"), (alpha, "Alpha Mask"), (foreground_alpha, "Foreground Alpha"), (new_bg_1, "New Sample Background")]]

    if task_type == "Upscale":
        # assert text_prompt, "Text prompt for image is required to Upscale! "
        if sd_ups_pipeline is None:
            # load model and scheduler
            sd_ups_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                SD_UPS_CHECKPOINT, revision="fp16", **sd_kwargs
            )
            sd_ups_pipeline = sd_ups_pipeline.to(device)
            if dpm_scheduler: sd_ups_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sd_ups_pipeline.scheduler.config)
            if attn_slicing: sd_ups_pipeline.enable_attention_slicing()
        
        ups_img = sd_ups_pipeline(prompt=sd_prompt, negative_prompt=negative_prompt, image=image_pil, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        return [caption, [(ups_img, "Stable Diffusion Upscaled Image")]]
    

    if task_type == "Text-to-Image":
        assert text_prompt, "Text prompt is required for Text-to-Image! "
        if sd_gen_pipeline is None:
            if sd_i2i_pipeline is not None:
                sd_gen_pipeline = StableDiffusionPipeline.from_pretrained(**sd_i2i_pipeline.components)
            else:
                sd_gen_pipeline = StableDiffusionPipeline.from_pretrained(SD_GEN_CHECKPOINT, **sd_kwargs)
            sd_gen_pipeline = sd_gen_pipeline.to(device)
            if dpm_scheduler: sd_gen_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sd_gen_pipeline.scheduler.config)
            if attn_slicing: sd_gen_pipeline.enable_attention_slicing()

        text2img = sd_gen_pipeline(prompt=sd_prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        # text2img = np.array(text2img)
        return [caption, [(text2img, "Stable Diffusion Generated Image")]]

    if task_type == "Drawing-to-Image":
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        scribble = scribble.convert("RGB")
        control_image = processor(scribble, scribble=True)

        if sd_cn_pipeline is None:
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_SCRIB_CKP, **sd_kwargs)
            sd_cn_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                SD_CNTRLNET_CKP, controlnet=controlnet, torch_dtype=torch.float16
            )

            sd_cn_pipeline.scheduler = UniPCMultistepScheduler.from_config(sd_cn_pipeline.scheduler.config)
            if attn_slicing: sd_cn_pipeline.enable_attention_slicing()
            sd_cn_pipeline.enable_model_cpu_offload()

        # generator = torch.manual_seed(0)
        draw_img = sd_cn_pipeline(sd_prompt, image=control_image, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        return [caption, [(draw_img, "Stable Diffusion ControlNet Generated Image")]]

    if task_type == "Image-to-Image":
        if sd_i2i_pipeline is None:
            if sd_gen_pipeline is not None:
                sd_i2i_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(**sd_gen_pipeline.components)
            else:
                sd_i2i_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(SD_I2I_CHECKPOINT, **sd_kwargs)
            sd_i2i_pipeline = sd_i2i_pipeline.to(device)
            if dpm_scheduler: sd_i2i_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sd_i2i_pipeline.scheduler.config)
            if attn_slicing: sd_i2i_pipeline.enable_attention_slicing()

        img2img = sd_i2i_pipeline(prompt=sd_prompt, negative_prompt=negative_prompt, image=image_pil, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        # text2img = np.array(text2img)
        return [caption, [(img2img, "Stable Diffusion Generated Image-to-Image")]]

    else:
        print("task_type:{} error!".format(task_type))
        return ["Error: Task type incorrect.", []]



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image Anything", add_help=True)
    parser.add_argument("--debug", action="store_true", help="use debug mode")
    parser.add_argument("--share", action="store_true", help="generate gradio share link")
    parser.add_argument('--port', type=int, default=7589, help="port to run the server")
    parser.add_argument('--no-gradio-queue', action="store_true", help="no gradio queue")
    args = parser.parse_args()

    print(args)

    block = gr.Blocks()
    if not args.no_gradio_queue:
        block = block.queue()

    with block:
        gr.Markdown("#  <center>Image Anything")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil",  tool="sketch")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Canvas for Drawing-to-Image task -------> ")
                        gr.Markdown("Click on brush icon to adjust brush size.")
                    with gr.Column():
                        create_button = gr.Button(label="Canvas", value="Open Canvas")
                    
                gr.Markdown("<h3>Image Caption:</h3>")
                img_caption = gr.Markdown()

                gallery = gr.Gallery(
                    label="Generated Images", show_label=False, elem_id="gallery"
                ).style(preview=True, object_fit="contain", height="auto")
                

            with gr.Column():        
                task_type = gr.Dropdown(["Image Caption", "Auto SAM Mask", "Detection/Annotation/Segmentation", "Remove/Replace Background", "Inpainting", "Upscale", "Text-to-Image", "Drawing-to-Image", "Image-to-Image"], value="Detection/Annotation/Segmentation", label="Select Task")
                gr.Markdown("Interaction Mode - Choose one: \n1. Click points on the image | \t2. Provide input text prompt | \t3. No interaction (auto mode)")
                text_prompt = gr.Textbox(label="Input Text Prompt <Specify objects to Detect/Segment/Annotate/Inpaint/Separate-Foreground>")
                sd_prompt = gr.Textbox(label="Stable Diffusion Prompt <Inpaint-target/Upscale/Text-to-image/Drawing-to-Image/Image-to-Image/New-Background>")
                # bg_prompt = gr.Textbox(label="Background Prompt <New SD background. Select Background Remove/Replace task >")
                negative_prompt = gr.Textbox(label="Negative Prompt for Stable Diffusion")
                run_button = gr.Button(label="Run")

                with gr.Accordion("Advanced Settings", open=False):
                    # with gr.Box():
                    gr.Markdown("Point click Settings")
                    scribble_mode = gr.Radio(["merge", "split"], value="split", label="scribble mode")

                    gr.Markdown("DINO BBox Settings")
                    box_threshold = gr.Slider(label="box threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                    text_threshold = gr.Slider(label="text threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                    iou_threshold = gr.Slider(label="IOU threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05)

                    gr.Markdown("VitMatte Trimap Settings")
                    erode_kernel_size = gr.Slider(minimum=1, maximum=30, step=1, value=10, label="erode kernel size")
                    dilate_kernel_size = gr.Slider(minimum=1, maximum=30, step=1, value=10, label="dilate kernel size")
                    
                    gr.Markdown("Transparency Settings")
                    tr_prompt = gr.Textbox(lines=1, value="glass.lens.crystal.diamond.bubble.bulb.web.grid", label="transparency input text")
                    tr_box_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.005, value=0.5, label="transparency box threshold")
                    tr_text_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.005, value=0.25, label="transparency text threshold")

                    gr.Markdown("Inpainting Settings")
                    inpaint_mode = gr.Radio(["merge", "split"], value="merge", label="inpaint mode")

                    gr.Markdown("Stable Diffusion Settings")
                    guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=7.5, label="Guidance Scale for prompt")
                    strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="Noising of reference image")
                    num_inference_steps = gr.Slider(minimum=1, maximum=60, step=1, value=30, label="Num of Inference Steps")

                    gr.Markdown("Model Settings")
                    model_matte = gr.Radio(['ViTMatte (Matte Anything)', 'Matte Anything Model (MAM)'], value='ViTMatte (Matte Anything)', label='Alpha Matte Model')
                    model_sam = gr.Radio(['SAM - Meta', 'Mobile SAM'], value='SAM - Meta', label='SAM Model')
                    # openai_api_key= gr.Textbox(label="(Optional)OpenAI key, enable chatgpt")

        input_image.upload(
            clear_old, [], []
            )
        create_button.click(
            fn=create_canvas, inputs=[], outputs=[input_image]
            )

        inputs=[input_image, task_type, text_prompt, sd_prompt, negative_prompt, scribble_mode, box_threshold, text_threshold, iou_threshold, erode_kernel_size, 
                dilate_kernel_size, tr_prompt, tr_box_threshold, tr_text_threshold, inpaint_mode, model_matte, model_sam, guidance_scale, strength, num_inference_steps]
        outputs=[img_caption, gallery]
        run_button.click(fn=run_image_anything, inputs=inputs, outputs=outputs)

    block.queue(concurrency_count=100)
    block.launch(server_name='0.0.0.0', server_port=args.port, debug=args.debug, share=args.share)