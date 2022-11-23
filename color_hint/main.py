#!/usr/bin/env python

import gradio as gr
import cv2.cv2 as cv

ORIGIN = None
HINT = None


def extract_mask(img):
  mask = (ORIGIN - img) != 0
  img = mask * img
  img = cv.resize(img, dsize=(256, 256), interpolation=cv.INTER_NEAREST)
  return img


def extract_hint(img):
  global HINT
  img = cv.resize(img, dsize=(16, 16), interpolation=cv.INTER_NEAREST)
  HINT = img
  img_view = cv.resize(img, dsize=(256, 256), interpolation=cv.INTER_NEAREST)
  return img_view


def save_loading(img):
  global ORIGIN
  if ORIGIN is None:
    ORIGIN = img


with gr.Blocks() as demo:
  with gr.Box():
    image = gr.Image(source="upload",
                     tool="color-sketch",
                     label="Canvas",
                     interactive=True)
    with gr.Row():
      with gr.Column():
        viewer_mask = gr.Image(interactive=False,
                               label="Mask",
                               shape=(256, 256)).style(height=400)
        btn_cvt2point = gr.Button("Convert to Point")
      with gr.Column():
        viewer_hint = gr.Image(interactive=False, label="Hint",
                               shape=(16, 16)).style(height=400)
        btn_cvt2hint = gr.Button("Convert to Hint")

  btn_estimate = gr.Button("Colorize!").style()

  viewer_result = gr.Image(interactive=False, shape=(256, 256),
                           label="Result").style(height=400)

  image.change(fn=save_loading, inputs=image, outputs=None)
  btn_cvt2point.click(fn=extract_mask, inputs=image, outputs=viewer_mask)
  btn_cvt2hint.click(fn=extract_hint, inputs=viewer_mask, outputs=viewer_hint)
  # btn_estimate.click()

demo.launch(share=False)
