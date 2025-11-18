import gradio as gr
import torch
import numpy as np
import supervision as sv
from typing import Iterable
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
import json
import ast
import re
from PIL import Image
from spaces import GPU

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "auto"

CATEGORIES = ["Query", "Caption", "Point", "Detect"]

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=DTYPE,
    device_map=DEVICE,
).eval()
qwen_processor = Qwen3VLProcessor.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
)

def safe_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def annotate_image(image: Image.Image, result: dict):
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image

    # Ensure image is mutable
    image = image.convert("RGB")
    original_width, original_height = image.size

    if "points" in result and result["points"]:
        points_list = [
            [int(p["x"] * original_width), int(p["y"] * original_height)]
            for p in result.get("points", [])
        ]
        if not points_list:
            return image

        points_array = np.array(points_list).reshape(1, -1, 2)
        key_points = sv.KeyPoints(xy=points_array)
        vertex_annotator = sv.VertexAnnotator(radius=4, color=sv.Color.RED)
        annotated_image = vertex_annotator.annotate(scene=np.array(image.copy()), key_points=key_points)
        return Image.fromarray(annotated_image)

    if "objects" in result and result["objects"]:
        boxes = []
        for obj in result["objects"]:
            x_min = obj.get("x_min", 0.0) * original_width
            y_min = obj.get("y_min", 0.0) * original_height
            x_max = obj.get("x_max", 0.0) * original_width
            y_max = obj.get("y_max", 0.0) * original_height
            boxes.append([x_min, y_min, x_max, y_max])

        if not boxes:
            return image

        detections = sv.Detections(xyxy=np.array(boxes))

        if len(detections) == 0:
            return image

        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
        annotated_image = box_annotator.annotate(scene=np.array(image.copy()), detections=detections)
        return Image.fromarray(annotated_image)

    return image

def run_qwen_inference(image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = qwen_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=512,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


@GPU
def process_qwen(image: Image.Image, category: str, prompt: str):
    if category == "Query":
        return run_qwen_inference(image, prompt), {}
    elif category == "Caption":
        full_prompt = f"Provide a {prompt} length caption for the image."
        return run_qwen_inference(image, full_prompt), {}
    elif category == "Point":
        full_prompt = (
            f"Provide 2d point coordinates for {prompt}. Report in JSON format."
        )
        output_text = run_qwen_inference(image, full_prompt)
        parsed_json = safe_parse_json(output_text)
        points_result = {"points": []}
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if "point_2d" in item and len(item["point_2d"]) == 2:
                    x, y = item["point_2d"]
                    points_result["points"].append({"x": x / 1000.0, "y": y / 1000.0})
        return json.dumps(points_result, indent=2), points_result
    elif category == "Detect":
        full_prompt = (
            f"Provide bounding box coordinates for {prompt}. Report in JSON format."
        )
        output_text = run_qwen_inference(image, full_prompt)
        parsed_json = safe_parse_json(output_text)
        objects_result = {"objects": []}
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                    xmin, ymin, xmax, ymax = item["bbox_2d"]
                    objects_result["objects"].append(
                        {
                            "x_min": xmin / 1000.0,
                            "y_min": ymin / 1000.0,
                            "x_max": xmax / 1000.0,
                            "y_max": ymax / 1000.0,
                        }
                    )
        return json.dumps(objects_result, indent=2), objects_result
    return "Invalid category", {}

def process_inputs(image, category, prompt):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please provide a prompt.")

    image.thumbnail((512, 512))

    qwen_text, qwen_data = process_qwen(image, category, prompt)
    qwen_annotated_image = annotate_image(image.copy(), qwen_data)

    return qwen_annotated_image, qwen_text

def on_category_change(category: str):
    if category == "Query":
        return gr.Textbox(placeholder="e.g., Count the total number of boats and describe the environment.")
    elif category == "Caption":
        return gr.Textbox(placeholder="e.g., short, normal, detailed")
    elif category == "Point":
        return gr.Textbox(placeholder="e.g., The gun held by the person.")
    elif category == "Detect":
        return gr.Textbox(placeholder="e.g., The headlight of the car.")
    return gr.Textbox(placeholder="e.g., detect the object.")


css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

with gr.Blocks(css=css, theme=steel_blue_theme) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Qwen-3VL: Multimodal Understanding**", elem_id="main-title")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                category_select = gr.Radio(
                    choices=CATEGORIES,
                    value="Query",
                    label="Select Task Category",
                    interactive=True,
                )
                prompt_input = gr.Textbox(
                    placeholder="e.g., Count the total number of boats and describe the environment.",
                    label="Prompt",
                    lines=1,
                )
                submit_btn = gr.Button("Process Image", variant="primary")

            with gr.Column(scale=2):
                qwen_img_output = gr.Image(label="Output Image")
                qwen_text_output = gr.Textbox(
                    label="Text Output", lines=10, interactive=False, show_copy_button=True
                )

        gr.Examples(
            examples=[
                ["examples/5.jpg", "Point", "Detect the children who are out of focus and wearing a white T-shirt."],
                ["examples/5.jpg", "Detect", "Point out the out-of-focus (all) children."],
                ["examples/4.jpg", "Detect", "Headlight"],
                ["examples/3.jpg", "Point", "Gun"],
                ["examples/1.jpg", "Query", "Count the total number of boats and describe the environment."],
                ["examples/2.jpg", "Caption", "a brief"],
            ],
            inputs=[image_input, category_select, prompt_input],
        )

    category_select.change(
        fn=on_category_change,
        inputs=[category_select],
        outputs=[prompt_input],
    )

    submit_btn.click(
        fn=process_inputs,
        inputs=[image_input, category_select, prompt_input],
        outputs=[qwen_img_output, qwen_text_output],
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True, ssr_mode=False, show_error=True)