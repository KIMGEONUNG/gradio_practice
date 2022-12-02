#!/usr/bin/env python

import os
import random

import gradio as gr
from pycomar.samples import get_img


def fake_gan():
    return [get_img(1), get_img(2)]

def add(x):
    print(x)
    print(x.keys())
    return x + [get_img(1)]

with gr.Blocks() as demo:
    with gr.Column(variant="panel"):
        with gr.Row():
            preview = gr.Image()
            ex = gr.Examples(examples=['sample01.jpg', 'sample02.jpg'], inputs=preview)
        btn = gr.Button("Add")
    # btn.click(add, inputs=ex, outputs=ex)

if __name__ == "__main__":
    demo.launch()
