#!/usr/bin/env python

from __future__ import annotations

import gradio as gr
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from typing import TypedDict
import numpy as np
from pathlib import Path
import PIL
import warnings
from gradio import processing_utils
from pycomar.samples import get_img

with gr.Blocks() as demo:
  print('demo is ', demo)
  view = gr.Image(interactive=True)
  gr.Examples(["sample01.jpg", "sample02.jpg"], inputs=view)
  btn = gr.Button("Start Training")
  im = get_img(1)
  demo.load(lambda : 1, input=None, outputs=im)

  def start_train():
    print('Training started')
    print(view)
    print('Training ended')
    pass

  btn.click(fn=start_train, inputs=None, outputs=None)
  demo.load

demo.launch()
