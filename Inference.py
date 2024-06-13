import os
import torch
from PIL import Image
from fsam.fastsam import FastSAM, FastSAMPrompt

def get_predefined_input(image, text_prompt):
    inputs = {}
    inputs["model_path"] = "./weights/FastSAM.pt"
    inputs["img_path"] = image
    inputs["imgsz"] = 1024
    inputs["iou"] = 0.9
    inputs["text_prompt"] = text_prompt or None  # Example: "a dog"
    inputs["conf"] = 0.4
    inputs["output"] = "./output/"
    inputs["better_quality"] = True

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
    input_image = Image.open(args["img_path"])
    input_image = input_image.convert("RGB")
    
    everything_results = model(
        input_image,
        device=args["device"],
        retina_masks=False,
        imgsz=args["imgsz"],
        conf=args["conf"],
        iou=args["iou"]
    )

    prompt_process = FastSAMPrompt(input_image, everything_results, device=args["device"])
    if args["text_prompt"]:
        ann = prompt_process.text_prompt(text=args["text_prompt"])
    else:
        ann = prompt_process.everything_prompt()
    
    # Create a valid output path
    image_filename = os.path.basename(args["img_path"])
    output_path = os.path.join(args["output"], image_filename)

    mask_path = prompt_process.plot(
        annotations=ann,
        output_path=output_path,
        withContours=args["withContours"],
        better_quality=args["better_quality"],
    )
    return mask_path

if __name__ == "__main__":
    image_path = "C:\\Users\\KIIT\\Desktop\\BITS Pilani Research Docs\\MODEL\\images\\dogs.jpg"  # Replace with your image path
    text_prompt = "a black dog"  # Replace with your text prompt
    args = get_predefined_input(image_path, text_prompt)
    output_mask_location = create_mask(args)
    print(f"Output mask saved at: {output_mask_location}")
