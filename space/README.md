---
title: Chest X-ray Nodule Classifier
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Flags lung nodules/masses on chest X-rays, with Grad-CAM
models:
  - trungvu0307/chest-xray-nodule-densenet121
---

# Chest X-ray Nodule Classifier: demo

Upload a frontal chest X-ray, or pick one of the test-set examples. The demo shows:
- the model's score
- whether the X-ray is flagged at the validation-chosen threshold of 0.120
- a Grad-CAM heatmap of the regions that raised the score

- **Model:** [trungvu0307/chest-xray-nodule-densenet121](https://huggingface.co/trungvu0307/chest-xray-nodule-densenet121)
- **Code and evaluation:** [GitHub](https://github.com/vutr2/chest-xray-nodule-classifier)
- **Example images:** NIH Clinical Center, ChestX-ray14 (Wang et al., CVPR 2017)

Not for clinical use.
