#!/usr/bin/env python

import numpy as np
import gradio as gr

img = None


def sepia(image):
  print(img)
  return image


def save_loading(input):
  print(input)
  img = input
  return img


with gr.Blocks() as demo:
# image = gr.ImagePaint(load_fn=save_loading)
  image = gr.Image(source="upload", tool="color-sketch", interactive=True)
  image.change(fn=save_loading, inputs="image", outputs="image")
  demo = gr.Interface(sepia, image, "image")
demo.launch()
