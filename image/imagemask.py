#!/usr/bin/env python

import numpy as np
import gradio as gr


def sepia(args):
  image = args["image"]
  mask = args["mask"][:, :, :1]
  mask = (mask / 255).astype('int')
  rs = image * mask
  return rs


demo = gr.Interface(sepia, gr.ImageMask(visible=True), "image")
demo.launch()
