#!/usr/bin/env python

import gradio as gr
from PIL import Image
import cv2.cv2 as cv
from random import sample

TOOGLE_UPLOAD = False
ORIGIN = None

samples = [
    Image.open('assets/sample1.jpg'),
    Image.open('assets/sample2.jpg'),
    Image.open('assets/sample3.jpg'),
    Image.open('assets/sample4.jpg'),
]


def inpainting():
  global samples
  results = sample(samples, 3)
  return results


if __name__ == "__main__":
  with gr.Blocks() as demo:
    gr.Markdown("# UI POC")
    btn_start = gr.Button("Stop")
    with gr.Box():
      masking = gr.ImageMask()
    btn_start.click(fn=lambda: Image.open('assets/sample2.jpg'),
                    outputs=[masking])

    btn_inpaint = gr.Button("Inpaint")
    with gr.Box():
      with gr.Row():
        with gr.Column():
          viewer1 = gr.Image(interactive=True, shape=(200, 200), type='pil')
          btn1 = gr.Button("Select")
        with gr.Column():
          viewer2 = gr.Image(interactive=True, shape=(200, 200), type='pil')
          btn2 = gr.Button("Select")
        with gr.Column():
          viewer3 = gr.Image(interactive=True, shape=(200, 200), type='pil')
          btn3 = gr.Button("Select")

      with gr.Box():
        gr.Markdown("Target")
        viewer4 = gr.Image(interactive=True, shape=(200, 200), type='pil')

    btn1.click(fn=lambda x: x, inputs=viewer1, outputs=viewer4)
    btn2.click(fn=lambda x: x, inputs=viewer2, outputs=viewer4)
    btn3.click(fn=lambda x: x, inputs=viewer3, outputs=viewer4)

    btn_inpaint.click(fn=inpainting,
                  inputs=None,
                  outputs=[viewer1, viewer2, viewer3])

    btn3 = gr.Button("Resume")
  demo.launch()
