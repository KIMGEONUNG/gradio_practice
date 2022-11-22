#!/usr/bin/env python

import numpy as np
import gradio as gr


def sepia(image):
  print(image)
  return image


demo = gr.Interface(sepia, gr.Webcam(), "image")
demo.launch()
