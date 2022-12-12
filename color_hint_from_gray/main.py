#!/usr/bin/env python

import gradio as gr
from skimage.color import rgb2gray
import numpy as np

def gen_mask(x: np.ndarray):
    # mask = (x[..., 0] == x[..., 1]) & (x[..., 1] == x[..., 2]) & (x[..., 2] == x[..., 0]) == False
    # x = mask[..., None] * x
    return x

with gr.Blocks() as demo:
    with gr.Row():
        view_color = gr.Image(label="Color")
        view_gray = gr.ImagePaint(label="Gray")
        view_mask = gr.Image(label="Mask")
    gr.Examples(examples=["sample01.jpg", "sample02.jpg"], inputs=view_color)

    # Events
    view_color.change(lambda x: rgb2gray(x), inputs=view_color, outputs=view_gray)
    # view_gray.change(gen_mask, inputs=view_gray, outputs=view_mask)

demo.launch(debug=True)
