#!/usr/bin/env python

import numpy as np
import gradio as gr

def sepia(image):
  return image

demo = gr.Interface(sepia, gr.Paint(), "image")
demo.launch()
