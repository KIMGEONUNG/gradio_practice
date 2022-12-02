#!/usr/bin/env python

import os
import random

import gradio as gr
from pycomar.samples import get_img

imgs = [get_img(1), get_img(2)]

def add(x):
    print(x)
    global imgs
    imgs += [get_img(1)]
    return imgs

with gr.Blocks() as demo:
    with gr.Column(variant="panel"):
        gallery = gr.Gallery(label="Dataset Preview",
                             show_label=True,
                             elem_id="gallery").style(grid=[3], height=300)
        btn_add = gr.Button("Add")
        btn_sel = gr.Button("Select")

    demo.load(lambda: [get_img(1), get_img(2)], inputs=None, outputs=gallery)
    print(gallery.click)
    btn_add.click(add, inputs=gallery, outputs=gallery)

if __name__ == "__main__":
    demo.launch()
