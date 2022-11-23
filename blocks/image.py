#!/usr/bin/env python

import gradio as gr

TOOGLE_UPLOAD = False
ORIGIN = None


def update(img):
  global ORIGIN
  mask = (img - ORIGIN) == 0
  mask = (1 - mask)
  img = mask * img
  return img


def event_edit(img):
  return None


def event_clear(img):
  return None


def event_change(img):
  global ORIGIN
  # global TOOGLE_UPLOA
  if ORIGIN is None:
    ORIGIN = img
    # TOOGLE_UPLOAD = False


def event_upload(arg):
  global ORIGIN
  ORIGIN = None
  return None


with gr.Blocks() as demo:
  gr.Markdown("Start typing below and then click **Run** to see the output.")
  with gr.Row():
    canvas1 = gr.ImagePaint(shape=(500, 500)).style(height=500)
    # canvas1.upload(fn=event_upload, inputs=canvas1)
    canvas1.change(fn=event_change, inputs=canvas1)
    # canvas1.edit(fn=event_edit, inputs=canvas1)
    # canvas1.clear(fn=event_clear, inputs=canvas1)
    viewer1 = gr.Image(interactive=False)

  btn = gr.Button("Extrac Color Mask")
  btn.click(fn=update, inputs=canvas1, outputs=viewer1)

demo.launch()
