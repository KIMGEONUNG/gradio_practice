#!/usr/bin/env python

import gradio as gr
from pycomar.samples import get_img


class Realtime(object):

  def __init__(self):
    self.cnt = 1
    self.imgs = [get_img(1), get_img(2)]
    self.on_training = True
    self.img_preview = get_img(2)

    with gr.Blocks() as self.demo:
      self.preview = gr.Image().style(height=300)
      with gr.Row():
        btn_start_train = gr.Button("Start")
        btn_stop_train = gr.Button("Stop")

      # EVENTS
      self.demo.load(self.update, inputs=None, outputs=self.preview, every=1)
      btn_start_train.click(self.on_toogle)
      btn_stop_train.click(self.off_toogle)

  def update(self):
    if self.on_training:
      self.cnt += 1
    return self.imgs[self.cnt % 2]

  def on_toogle(self):
    self.on_training = True

  def off_toogle(self):
    self.on_training = False

  def launch(self):
    self.demo.queue().launch()


if __name__ == "__main__":
  gui = Realtime()
  gui.launch()
