import os
import random

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
# import GroundingDINO.groundingdino.datasets.transforms as T
# from GroundingDINO.groundingdino.models import build_model
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import sys
sys.path.insert(0, './GroundingDINO')
from groundingdino.util.inference import Model
import groundingdino.datasets.transforms as T

# segment anything
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np

# diffusers
import torch
from diffusers import StableDiffusionInpaintPipeline

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

# Matte Anything (VitMatte)
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# ckpt_repo_id = "ShilongLiu/GroundingDINO"
GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swint_ogc.pth"
SAM_MODEL = 'vit_h'
SAM_CHECKPOINT="./checkpoints/sam_vit_h_4b8939.pth"
output_dir="outputs"
device="cuda"

vitmatte_models = {
	'vit_b': './checkpoints/ViTMatte_B_DIS.pth',
}

vitmatte_config = {
	'vit_b': 'Matte_Anything/configs/matte_anything.py',
}

GR_PALETTE = (51, 255, 146)

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
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# def transform_image(image_pil):

#     transform = T.Compose(
#         [
#             T.RandomResize([800], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image, _ = transform(image_pil, None)  # 3, h, w
#     return image


# def load_model(model_config_path, model_checkpoint_path, device):
#     args = SLConfig.fromfile(model_config_path)
#     args.device = device
#     model = build_model(args)
#     checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#     load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     print(load_res)
#     _ = model.eval()
#     return model

# def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
#     caption = caption.lower()
#     caption = caption.strip()
#     if not caption.endswith("."):
#         caption = caption + "."

#     with torch.no_grad():
#         outputs = model(image[None], captions=[caption])
#     logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
#     boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
#     logits.shape[0]

#     # filter output
#     logits_filt = logits.clone()
#     boxes_filt = boxes.clone()
#     filt_mask = logits_filt.max(dim=1)[0] > box_threshold
#     logits_filt = logits_filt[filt_mask]  # num_filt, 256
#     boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
#     logits_filt.shape[0]

#     # get phrase
#     tokenlizer = model.tokenizer
#     tokenized = tokenlizer(caption)
#     # build pred
#     pred_phrases = []
#     scores = []
#     for logit, box in zip(logits_filt, boxes_filt):
#         pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
#         if with_logits:
#             pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
#         else:
#             pred_phrases.append(pred_phrase)
#         scores.append(logit.max().item())

#     return boxes_filt, torch.Tensor(scores), pred_phrases

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

    cfg = LazyConfig.load(vitmatte_config[model_type])
    vitmatte = instantiate(cfg.model)
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(vitmatte_models[model_type])

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
    return None



blip_processor = None
blip_model = None
groundingdino_model = None
sam_predictor = None
sam_automask_generator = None
sd_pipeline = None
vitmatte = None
caption = None


def run_grounded_sam(input_image, task_type, text_prompt, inpaint_prompt, bg_prompt, scribble_mode, box_threshold, text_threshold, iou_threshold, 
                                                      erode_kernel_size, dilate_kernel_size, tr_prompt, tr_box_threshold, tr_text_threshold, inpaint_mode, model_matte, model_sam):
    
    global blip_processor, blip_model, groundingdino_model, sam_predictor, sam_automask_generator, sd_pipeline, vitmatte, caption
    # anns, mask, alpha, vitmatte, new_bg_sample, new_bg_diffusion, inpaint = None, None, None, None, None, None, None
    if caption == 'markdown':
        caption = None

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image = input_image["image"]
    scribble = input_image["mask"]
    size = image.size # w, h
    image_pil = image.convert("RGB")
    image = np.array(image_pil)
    # transformed_image = transform_image(image_pil)
    

    if sam_predictor is None:
        # initialize SAM
        assert SAM_CHECKPOINT, 'SAM_CHECKPOINT is not found!'
        # sam = build_sam(checkpoint=SAM_CHECKPOINT)
        # sam.to(device=device)
        sam = sam_model_registry[SAM_MODEL](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)
        sam.eval()
        sam_predictor = SamPredictor(sam)
        sam_automask_generator = SamAutomaticMaskGenerator(sam)

    if groundingdino_model is None:
        groundingdino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=device)


    if task_type == "Image Caption":
        print(f"Caption b4: {caption}")
        # if caption is None:
            # generate caption and tags
            # use Tag2Text can generate better captions
            # https://huggingface.co/spaces/xinyu1205/Tag2Text
            # but there are some bugs...
        blip_processor = blip_processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = blip_model or BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        caption = generate_caption(blip_processor, blip_model, image_pil)
            # caption = text_prompt
            # if len(openai_api_key) > 0:
            #     text_prompt = generate_tags(text_prompt, split=",", openai_api_key=openai_api_key)
        print(f"Caption after: {caption}")
        return [caption, []]

    if task_type == "Auto SAM Mask":
        masks = sam_automask_generator.generate(image)
        # mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
        # mask_draw = ImageDraw.Draw(mask_image)
        # for mask in masks:
        #     draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

        # image_pil = image_pil.convert('RGBA')
        # image_pil.alpha_composite(mask_image)
        # (image_pil, "Image with SAM Mask"), (mask_image, "SAM Mask"),

        full_img, res = show_anns(masks)
        return [caption, [ (full_img, "show_anns fn full_img")]]

    if task_type == "Detection/Annotation/Segmentation" or task_type == "Inpainting" or task_type == "Remove/Replace Background":
        sam_predictor.set_image(image)
        point_coords, point_labels, transformed_boxes = None, None, None
        print("scribble=======", np.max(scribble))
        # return [caption, [(scribble, "Scribble")]]
        scribble = scribble.convert("RGB")
        scribble = np.array(scribble)
        scribble = scribble.transpose(2, 1, 0)[0]

        # User selected regions (circle/disk)
        labeled_array, num_features = ndimage.label(scribble >= 255)
        
        print("num-features========== ", num_features)

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

        else:
            if text_prompt == "" or text_prompt is None:
                if caption == "" or caption is None:
                    blip_processor = blip_processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                    blip_model = blip_model or BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
                    caption = generate_caption(blip_processor, blip_model, image_pil)

                text_prompt = caption
            
            # run grounding dino model
            detections, pred_phrases = groundingdino_model.predict_with_caption(
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                caption = text_prompt, 
                box_threshold = box_threshold, 
                text_threshold = text_threshold
            )
            # # process boxes
            # H, W = size[1], size[0]
            # for i in range(boxes_filt.size(0)):
            #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            #     boxes_filt[i][2:] += boxes_filt[i][:2]

            # boxes_filt = boxes_filt.cpu()

            # use NMS to handle overlapped boxes
            # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            # nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            # boxes_filt = boxes_filt[nms_idx]
            # pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            # print(f"After NMS: {boxes_filt.shape[0]} boxes")
            # print(f"Revise caption with number: {text_prompt}")
            # use NMS to handle overlapped boxes
            if len(detections.xyxy) > 1:
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), 
                    torch.from_numpy(detections.confidence), 
                    iou_threshold,
                ).numpy().tolist()

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
            
            transformed_boxes = sam_predictor.transform.apply_boxes(detections.xyxy, image.shape[:2])
            transformed_boxes = torch.as_tensor(transformed_boxes, dtype=torch.float).to(device)

        masks, _, _ = sam_predictor.predict_torch(
                point_coords = point_coords,
                point_labels = point_labels,
                boxes = transformed_boxes,
                multimask_output = False,
            )

        if task_type == "Detection/Annotation/Segmentation":
            if num_features < 1:
                image_draw = ImageDraw.Draw(image_pil)

                for box, label in zip(detections.xyxy, pred_phrases):
                    draw_box(box, image_draw, label)

            mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)
            for mask in masks:
                draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

            image_pil = image_pil.convert('RGBA')
            image_pil.alpha_composite(mask_image)

            # anns = image_pil
            # mask = mask_image
            return [caption, [(image_pil, "Annotations/Segmentation"), (mask_image, "SAM Mask")]]        
            

        if task_type == "Inpainting":
            assert inpaint_prompt, 'inpaint_prompt is not found!'
            # inpainting pipeline
            if inpaint_mode == 'merge':
                masks = torch.sum(masks, dim=0).unsqueeze(0)
                masks = torch.where(masks > 0, True, False)
            mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
            mask_pil = Image.fromarray(mask)
            
            if sd_pipeline is None:
                sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
                )
                sd_pipeline = sd_pipeline.to("cuda")

            image = sd_pipeline(prompt=inpaint_prompt, image=image_pil.resize((512, 512)), mask_image=mask_pil.resize((512, 512))).images[0]
            image = image.resize(size)

            # mask = mask_pil
            # inpaint = image
            return [caption, [(image, "Inpainting"), (mask_pil, "SAM Mask")]]


        if task_type == "Remove/Replace Background":

            if vitmatte is None:
                vitmatte = init_vitmatte('vit_b')
            masks = masks.cpu().detach().numpy()
            # mask_all = np.ones((image.shape[0], image.shape[1], 3))
            # for ann in masks:
            #     color_mask = np.random.random((1, 3)).tolist()[0]
            #     for i in range(3):
            #         mask_all[ann[0] == True, i] = color_mask[i]
            # img = image / 255 * 0.3 + mask_all * 0.7

            # generate alpha matte
            torch.cuda.empty_cache()
            mask = np.zeros(masks[0][0].shape)
            print("Mask shape ", mask.shape, masks.shape[0])
            for i in range(masks.shape[0]):
                mask[masks[i][0] == True] = 255
            mask = mask.astype(np.uint8)
            print("Trimap in values: ", np.unique(mask))
            trimap = generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
            print("Trimap out values: ", np.unique(trimap))
            trimap[trimap==128] = 0.5
            trimap[trimap==255] = 1

            # boxes, logits, phrases = dino_predict(
            #     model=grounding_dino,
            #     image=image_transformed,
            #     caption=tr_caption,
            #     box_threshold=tr_box_threshold,
            #     text_threshold=tr_text_threshold
            #     )
            
            # # run grounding dino model
            # boxes, scores, phrases = get_grounding_output(
            #     groundingdino_model, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), tr_prompt, tr_box_threshold, tr_text_threshold
            # )
            # run grounding dino model
            boxes, phrases = groundingdino_model.predict_with_caption(
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                caption = tr_prompt, 
                box_threshold = tr_box_threshold, 
                text_threshold = tr_text_threshold
            )

            # annotated_frame = dino_annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
            
            # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

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
            # background = generate_checkerboard_image(image.shape[0], image.shape[1], 8)
            background = np.array([GR_PALETTE], dtype='uint8')

            # calculate foreground with alpha blending
            foreground_alpha = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

            # calculate foreground with mask
            # foreground_mask = image * np.expand_dims(mask/255, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(mask/255, axis=2).repeat(3,2))/255

            foreground_alpha[foreground_alpha>1] = 1
            # foreground_mask[foreground_mask>1] = 1

            # return img, mask_all
            trimap[trimap==1] == 0.999

            # new background

            background_1 = cv2.imread('assets/sea.jpeg')
            # background_2 = cv2.imread('figs/forest.jpg')
            # background_3 = cv2.imread('figs/sunny.jpg')

            background_1 = cv2.resize(background_1, (image.shape[1], image.shape[0]))
            # background_2 = cv2.resize(background_2, (image.shape[1], image.shape[0]))
            # background_3 = cv2.resize(background_3, (image.shape[1], image.shape[0]))

            # to RGB
            background_1 = cv2.cvtColor(background_1, cv2.COLOR_BGR2RGB)
            # background_2 = cv2.cvtColor(background_2, cv2.COLOR_BGR2RGB)
            # background_3 = cv2.cvtColor(background_3, cv2.COLOR_BGR2RGB)

            # use alpha blending
            new_bg_1 = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_1 * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255
            # new_bg_2 = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_2 * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255
            # new_bg_3 = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_3 * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

            if bg_prompt is not None:
                if sd_pipeline is None:
                    sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
                    )
                    sd_pipeline = sd_pipeline.to("cuda")

                background_img = sd_pipeline(bg_prompt).images[0]
                background_img = np.array(background_img)
                background_img = cv2.resize(background_img, (image.shape[1], image.shape[0]))
                new_bg_sd = image * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_img * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255
                return [caption, [(mask, "SAM Mask"), (alpha, "Alpha Mask"), (foreground_alpha, "Foreground Alpha"), (new_bg_1, "New Sample Background"), (new_bg_sd, "New Diffusion Background")]]
            else:
                return [caption, [(mask, "SAM Mask"), (alpha, "Alpha Mask"), (foreground_alpha, "Foreground Alpha"), (new_bg_1, "New Sample Background")]]

    if task_type == "Upscale":
        return [caption, []]
    if task_type == "Text-to-Image":
        return [caption, []]

    else:
        print("task_type:{} error!".format(task_type))

    # return [anns, mask, alpha, vitmatte, new_bg_sample, new_bg_diffusion, inpaint, upscale]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument('--port', type=int, default=7589, help='port to run the server')
    parser.add_argument('--no-gradio-queue', action="store_true", help='no gradio queue')
    args = parser.parse_args()

    print(args)

    block = gr.Blocks()
    if not args.no_gradio_queue:
        block = block.queue()

    with block:
        gr.Markdown("#<center>Image Anything")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil",  tool="sketch")
                # image_caption = caption
                gr.Markdown("<h3>Image Caption:</h3>")
                img_caption = gr.Markdown()
                # gr.Textbox(value= caption, label="Image Caption", interactive=False)
                # gr.Markdown(caption)
                gallery = gr.Gallery(
                    label="Generated Images", show_label=False, elem_id="gallery"
                ).style(preview=True, object_fit="contain", height="auto")
                

                # with gr.Tab(label='Annotations'):
                #     anns = gr.Image(type='numpy')
                # # show the image with mask
                # with gr.Tab(label='SAM Mask'):
                #     mask = gr.Image(type='numpy')
                # # with gr.Tab(label='Trimap'):
                # #     trimap = gr.Image(type='numpy')
                # with gr.Tab(label='Alpha Matte'):
                #     alpha = gr.Image(type='numpy')
                # # show only mask
                # # with gr.Tab(label='Foreground by SAM Mask'):
                # #     foreground_by_sam_mask = gr.Image(type='numpy')
                # with gr.Tab(label='Foreground'):
                #     vitmatte = gr.Image(type='numpy')
                # # with gr.Tab(label='Transparency Detection'):
                # #     transparency = gr.Image(type='numpy')
                # with gr.Tab(label='New Sample Background'):
                #     new_bg_sample = gr.Image(type='numpy')
                # with gr.Tab(label='New Diffusion Background'):
                #     new_bg_diffusion = gr.Image(type='numpy')
                # # with gr.Tab(label='New Background 3'):
                # #     new_bg_3 = gr.Image(type='numpy')
                # with gr.Tab(label='Inpainting'):
                #     inpaint = gr.Image(type='numpy')
                # with gr.Tab(label='Upscale'):
                #     upscale = gr.Image(type='numpy')

            with gr.Column():        
                task_type = gr.Dropdown(["Image Caption", "Auto SAM Mask", "Detection/Annotation/Segmentation", "Remove/Replace Background", "Inpainting", "Upscale", "Text-to-Image"], value="Detection/Annotation/Segmentation", label="Select Task")
                gr.Markdown("Interaction Mode - Choose one: \n1. Click points on the image | \t2. Provide input text prompt | \t3. No interaction (default mode)")
                text_prompt = gr.Textbox(label="Input Text Prompt")
                inpaint_prompt = gr.Textbox(label="Inpaint Prompt <To inpaint the mask with Stable Diffusion>")
                bg_prompt = gr.Textbox(label="Background Prompt <To generate new background with Stable Diffusion>")
                run_button = gr.Button(label="Run")

                with gr.Accordion("Advanced options", open=False):
                    # with gr.Box():
                    gr.Markdown("Point click Settings")
                    scribble_mode = gr.Dropdown(["merge", "split"], value="split", label="scribble mode")

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
                    inpaint_mode = gr.Dropdown(["merge", "first"], value="merge", label="inpaint mode")

                    gr.Markdown("Model Settings")
                    model_matte = gr.Radio(['ViTMatte (Matte Anything)', 'Matte Anything Model (MAM)'], value='ViTMatte (Matte Anything)', label='Alpha Matte Model')
                    model_sam = gr.Radio(['SAM - Meta', 'Mobile SAM'], value='SAM - Meta', label='SAM Model')
                    # openai_api_key= gr.Textbox(label="(Optional)OpenAI key, enable chatgpt")
                    # with gr.Box():
                    # gr.Markdown("Input Text Settings")
                    # fg_box_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, value=0.25, label="foreground_box_threshold")
                    # fg_text_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, value=0.25, label="foreground_text_threshold")

                    # with gr.Box():

        # input_image.upload(
        #     clear_old,
        #     [],
        #     [caption]
        # )
        run_button.click(fn=run_grounded_sam, inputs=[input_image, task_type, text_prompt, inpaint_prompt, bg_prompt, scribble_mode, box_threshold, text_threshold, iou_threshold, 
                                                      erode_kernel_size, dilate_kernel_size, tr_prompt, tr_box_threshold, tr_text_threshold, inpaint_mode, model_matte, model_sam], outputs=[img_caption, gallery])

    block.queue(concurrency_count=100)
    block.launch(server_name='0.0.0.0', server_port=args.port, debug=args.debug, share=args.share)