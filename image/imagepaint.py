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



demo = gr.Interface(sepia, gr.ImagePaint(load_fn=save_loading), "image")
demo.launch()
