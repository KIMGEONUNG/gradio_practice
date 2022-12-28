#!/usr/bin/env python

import gradio as gr
import torch

x = torch.randint(0, 10, (16, 16)).numpy()


def callback(arg):
    print(type(arg))
    print(arg)
    print(arg.to_numpy())


with gr.Blocks() as demo:
    with gr.Column():
        dataframe = gr.Dataframe(
            value=x,
            headers=['_'] * 16,
            interactive=True,
            row_count=(16, "fixed"),
            col_count=(16, "fixed"),
        )
        with gr.Row():
            btn = gr.Button("Set all")
            target_id = gr.Number()
    dataframe.change(callback, inputs=dataframe)
    btn.click(lambda x: [[x] * 16] * 16, inputs=target_id, outputs=dataframe)

demo.launch()
