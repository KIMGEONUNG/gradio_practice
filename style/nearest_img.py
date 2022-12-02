#!/usr/bin/env python

import gradio as gr
from PIL import Image
from torchvision.transforms import Resize

def masking(arg):
    mask = arg["mask"][..., :3]
    mask = Image.fromarray(mask)
    mask = Resize(16)(mask)
    return mask

css = "img {image-rendering: pixelated;}"

with gr.Blocks(css=css) as demo:
    btn = gr.Button("start")
    with gr.Row():
        mask = gr.ImageMask()
        paint = gr.ImagePaint()
    with gr.Row():
        view = gr.Image(interactive=False).style(height=500)

    btn.click(masking, inputs=mask, outputs=view)

demo.launch()
