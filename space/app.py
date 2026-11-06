"""Gradio demo: flags lung nodules/masses on chest X-rays and shows a Grad-CAM heatmap.

Runs on a ZeroGPU Space, but the model is small enough for CPU: inference happens outside
@spaces.GPU, so no GPU quota is used. ZeroGPU only requires one decorated function to exist.
"""
import json
from pathlib import Path

import spaces  # must be imported before torch on ZeroGPU

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models
from torchvision.transforms import v2 as T

MODEL_REPO = "trungvu0307/chest-xray-nodule-densenet121"
GITHUB_URL = "https://github.com/vutr2/chest_xray_classifier"
EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLE_LABELS = json.loads((EXAMPLES_DIR / "labels.json").read_text())   # file name -> NIH label

# ---------- model ----------
ckpt = torch.load(hf_hub_download(MODEL_REPO, "model.pt"), map_location="cpu")
THRESHOLD, IMG_SIZE = ckpt["threshold"], ckpt["img_size"]

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model.load_state_dict(ckpt["state_dict"])
model.eval()
cam = GradCAM(model=model, target_layers=[model.features[-1]])

preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@spaces.GPU(duration=1)
def _noop():  # ZeroGPU refuses to start without at least one decorated function; never called
    pass


OUTCOMES = {  # (NIH label has nodule/mass, model flagged) -> text
    (True, True): "✅ correct detection",
    (True, False): "❌ missed",
    (False, True): "⚠️ false alarm",
    (False, False): "✅ correctly not flagged",
}


def analyse(image_path: str | None) -> tuple[str, Image.Image | None]:
    """Score a frontal chest X-ray for lung nodule/mass and return a verdict plus a Grad-CAM overlay.

    Args:
        image_path: path to a chest X-ray image (PNG or JPEG, PA or AP view).

    Returns:
        Markdown with the score, the flag decision at the validation-chosen threshold and, for the
        bundled examples, the NIH label; and a 512x512 Grad-CAM overlay image.
    """
    if image_path is None:
        return "Upload a chest X-ray or pick an example.", None

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]
    base = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255
    overlay = Image.fromarray(show_cam_on_image(base, heat, use_rgb=True)).resize((512, 512), Image.BICUBIC)

    flagged = score >= THRESHOLD
    lines = [
        "### 🟠 Flagged: possible nodule or mass" if flagged else "### 🟢 Not flagged",
        f"Model score **{score:.3f}** · flag threshold **{THRESHOLD:.3f}**",
    ]
    truth = EXAMPLE_LABELS.get(Path(image_path).name)
    if truth:
        has_finding = truth["label"] != "No Finding"
        lines.append(f"NIH label for this example: **{truth['label']}** → {OUTCOMES[(has_finding, flagged)]}")
    return "\n\n".join(lines), overlay


INTRO = """
# 🫁 Chest X-ray Nodule Classifier

A DenseNet-121 fine-tuned on NIH ChestX-ray14 to flag **lung nodules and masses**. Upload a frontal chest X-ray or pick an example below.

**On held-out test patients:** AUC 0.812 · catches 90% of nodules/masses · but also flags about 55% of normal X-rays.
The threshold is low on purpose: for a screening aid, missing a nodule costs more than a false alarm.
"""

FOOTER = f"""
**Not for clinical use.** Research and educational demo.
Model: [Hugging Face](https://huggingface.co/{MODEL_REPO}) · Code and full evaluation: [GitHub]({GITHUB_URL}) ·
Example images: NIH Clinical Center, ChestX-ray14 (Wang et al., CVPR 2017).
"""

with gr.Blocks(title="Chest X-ray Nodule Classifier") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Chest X-ray (frontal view)", height=420)
            button = gr.Button("Analyse", variant="primary")
        with gr.Column():
            result = gr.Markdown()
            heatmap = gr.Image(label="Grad-CAM: regions that raised the score (most meaningful when flagged)",
                               height=420)
    gr.Examples(
        examples=[[str(EXAMPLES_DIR / name)] for name in EXAMPLE_LABELS],
        example_labels=[f"NIH label: {v['label']}" for v in EXAMPLE_LABELS.values()],
        inputs=image,
        outputs=[result, heatmap],
        fn=analyse,
        cache_examples=True,
        cache_mode="lazy",
        label="Examples from the test set (patients never seen in training)",
    )
    gr.Markdown(FOOTER)
    button.click(analyse, inputs=image, outputs=[result, heatmap])

if __name__ == "__main__":
    demo.launch()
