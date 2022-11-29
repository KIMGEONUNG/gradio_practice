#!/usr/bin/env python

import gradio as gr
import datetime
with gr.Blocks() as demo:
  with gr.Accordion("See Details"):
    gr.Markdown("lorem ipsum")

  def get_time():
    return datetime.datetime.now().time()

  dt = gr.Textbox(label="Current time")
  demo.load(get_time, inputs=None, outputs=dt, every=1)
demo.queue().launch()
