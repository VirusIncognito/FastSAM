import ast
import os
import torch
from PIL import Image
from fsam.fastsam import FastSAM, FastSAMPrompt
from fsam.utils.tools import convert_box_xywh_to_xyxy
import clip

def get_predefined_input(image, text_prompt):
    inputs = {}
    inputs["model_path"] = "./fsam/weights/FastSAM.pt"
    inputs["img_path"] = image
    inputs["imgsz"] = 1024
    inputs["iou"] = 0.9
    inputs["text_prompt"] = text_prompt or None  # Example: "a dog"
    inputs["conf"] = 0.4
    inputs["output"] = "./output/"
    inputs["randomcolor"] = True
    inputs["point_prompt"] = "[[0,0]]"
    inputs["point_label"] = "[0]"
    inputs["box_prompt"] = "[[0,0,0,0]]"
    inputs["better_quality"] = False

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    inputs["device"] = str(device)
    inputs["retina"] = True
    inputs["withContours"] = False
    
    return inputs

def create_mask(args):
    # load model
    model = FastSAM(args["model_path"])
    args["point_prompt"] = ast.literal_eval(args["point_prompt"])
    args["box_prompt"] = convert_box_xywh_to_xyxy(ast.literal_eval(args["box_prompt"]))
    args["point_label"] = ast.literal_eval(args["point_label"])
    input_image = Image.open(args["img_path"])
    input_image = input_image.convert("RGB")
    
    everything_results = model(
        input_image,
        device=args["device"],
        retina_masks=args["retina"],
        imgsz=args["imgsz"],
        conf=args["conf"],
        iou=args["iou"]
    )
    
    bboxes = None
    points = None
    point_label = None
    
    prompt_process = FastSAMPrompt(input_image, everything_results, device=args["device"])
    
    if args["box_prompt"][0][2] != 0 and args["box_prompt"][0][3] != 0:
        ann = prompt_process.box_prompt(bboxes=args["box_prompt"])
        bboxes = args["box_prompt"]
    elif args["text_prompt"]:
        ann = prompt_process.text_prompt(text=args["text_prompt"])
    elif args["point_prompt"][0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=args["point_prompt"], pointlabel=args["point_label"]
        )
        points = args["point_prompt"]
        point_label = args["point_label"]
    else:
        ann = prompt_process.everything_prompt()
    
    # Create a valid output path
    image_filename = os.path.basename(args["img_path"])
    output_path = os.path.join(args["output"], image_filename)

    prompt_process.plot(
        annotations=ann,
        output_path=output_path,
        bboxes=bboxes,
        points=points,
        point_label=point_label,
        withContours=args["withContours"],
        better_quality=args["better_quality"],
    )
    
    return output_path

if __name__ == "__main__":
    image_path = "C:\\Users\\KIIT\\Desktop\\BITS Pilani Research Docs\\MODEL\\fsam\\input\\image.jpg"  # Replace with your image path
    text_prompt = "a dog"  # Replace with your text prompt
    args = get_predefined_input(image_path, text_prompt)
    output_mask_location = create_mask(args)
    print(f"Output mask saved at: {output_mask_location}")
