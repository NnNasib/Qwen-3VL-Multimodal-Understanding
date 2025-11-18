# **Qwen-3VL: Multimodal Understanding**

> This Gradio-based web application leverages the **Qwen3-VL-4B-Instruct** model from Alibaba's Qwen series for multimodal tasks involving images and text. It enables users to upload an image and perform various vision-language tasks, such as querying details, generating captions, detecting points of interest, or identifying bounding boxes for objects. The app includes visual annotations for point and detection tasks using the `supervision` library.
Powered by Hugging Face Transformers, PyTorch, and Gradio, this demo showcases the model's capabilities in real-time image understanding.

<img width="1918" height="1223" alt="Screenshot 2025-11-18 at 17-20-08 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/14e0b157-3dec-410e-8fc6-4161210ab1e9" />
<img width="1918" height="1127" alt="Screenshot 2025-11-18 at 17-04-03 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/076a518a-25a0-4e8a-bc0b-746a34a6b936" />
<img width="1918" height="1206" alt="Screenshot 2025-11-18 at 18-04-14 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/bf0d08d4-b819-4b23-823d-77f7c83decd1" />
<img width="1918" height="1233" alt="Screenshot 2025-11-18 at 17-29-34 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/d8a663ea-1be9-486f-8fb6-84cc228c577a" />
<img width="1916" height="1230" alt="Screenshot 2025-11-18 at 17-24-33 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/cb216bb6-0be6-40b3-b9d7-67377bc24d0d" />

## Features

- **Query**: Ask open-ended questions about the image (e.g., "Count the total number of boats and describe the environment.").
- **Caption**: Generate image captions of varying lengths (e.g., short, detailed).
- **Point**: Detect and annotate 2D keypoints for specific elements (e.g., "The gun held by the person.").
- **Detect**: Identify and annotate bounding boxes for objects (e.g., "The headlight of the car.").
- **Visual Annotations**: Automatically overlays keypoints (red dots) or bounding boxes on the output image.
- **Custom Theme**: Steel-blue themed interface for a modern look.
- **Examples**: Pre-loaded sample images and prompts to get started quickly.
- **GPU Acceleration**: Optimized for CUDA devices if available.

The app processes images at a thumbnail resolution (512x512) for efficiency and supports JSON-formatted outputs for structured tasks.

## Demo

Try the live demo on Hugging Face Spaces:  
[https://huggingface.co/spaces/prithivMLmods/Qwen3-VL-HF-Demo](https://huggingface.co/spaces/prithivMLmods/Qwen3-VL-HF-Demo)

## Installation

### Prerequisites
- Python 3.10+ (recommended)
- pip >= 23.0.0

### Requirements
Install the dependencies using the provided `requirements.txt` or run the following:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1; adjust for your setup
pip install transformers==4.57.0
pip install supervision==0.26.1
pip install accelerate==1.10.1
pip install Pillow==11.3.0
pip install gradio==5.49.1
```

For a full list, see `requirements.txt`:
```
transformers==4.57.0
torchvision==0.23.0
supervision==0.26.1
accelerate==1.10.1
Pillow==11.3.0
gradio==5.49.1
torch==2.8.0
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen-3VL-Multimodal-Understanding.git
   cd Qwen-3VL-Multimodal-Understanding
   ```
2. Install dependencies (as above).
3. Download model weights automatically on first run (requires internet).

## Usage

1. Run the app locally:
   ```bash
   python app.py
   ```
   This launches a Gradio interface at `http://127.0.0.1:7860`.

2. In the interface:
   - Upload an image.
   - Select a task category (Query, Caption, Point, Detect).
   - Enter a prompt tailored to the category.
   - Click "Process Image" to generate results.

3. Outputs:
   - **Text**: Generated response or JSON (with copy button).
   - **Image**: Annotated version if applicable (points or boxes).

### Example Prompts
| Category | Example Prompt | Expected Output |
|----------|----------------|-----------------|
| Query | "Count the total number of boats and describe the environment." | Descriptive text with counts. |
| Caption | "detailed" | A long, descriptive caption. |
| Point | "The gun held by the person." | JSON with normalized (0-1) coordinates; red dots on image. |
| Detect | "Headlight of the car." | JSON with bounding boxes; colored rectangles on image. |

Sample images are included in the `examples/` folder for testing.

## Model Details
- **Model**: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (4B parameters, vision-language model).
- **Processor**: Handles chat templating and tokenization.
- **Device**: Auto-detects CUDA; falls back to CPU.
- **Limitations**: 
  - Max new tokens: 512.
  - Coordinates normalized to [0, 1000] in model output, scaled to [0, 1] in app.
  - No fine-tuning; relies on zero-shot prompting.

## Contributing
Feel free to fork the repo, submit issues, or pull requests. Contributions for new tasks, themes, or optimizations are welcome!

## Repository
- GitHub: [https://github.com/PRITHIVSAKTHIUR/Qwen-3VL-Multimodal-Understanding](https://github.com/PRITHIVSAKTHIUR/Qwen-3VL-Multimodal-Understanding)
- Hugging Face Space: [https://huggingface.co/spaces/prithivMLmods/Qwen3-VL-HF-Demo](https://huggingface.co/spaces/prithivMLmods/Qwen3-VL-HF-Demo)
