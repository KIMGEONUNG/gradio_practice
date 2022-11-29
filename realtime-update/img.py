#!/usr/bin/env python

import gradio as gr
import datetime
from pycomar.samples import get_img
with gr.Blocks() as demo:
  cnt = 1
  imgs = [get_img(1), get_img(2)]

  def get_time():
    global cnt
    cnt += 1
    return imgs[cnt % 2]

  dt = gr.Image()
  demo.load(get_time, inputs=None, outputs=dt, every=1)
demo.queue().launch()
