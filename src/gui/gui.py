
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import gradio as gr
from torchvision import transforms
from scipy.io import loadmat

from src.models.lit_module import LitModule

cwd = os.getcwd()

model = LitModule.load_from_checkpoint(f"{cwd}/final_checkpoint/last.ckpt", map_location=torch.device('cpu'))
model.eval()

labels = loadmat(f'{cwd}/data/stanford-cars-dataset/cars_annos.mat')['class_names'][0]

example_images = [f'{cwd}/src/gui/0000{i}.jpg' for i in range(1, 10)]



def load_model(model_type):
    print(model_type)
    model = LitModule.load_from_checkpoint(f"{cwd}/final_checkpoint/last.ckpt", map_location=torch.device('cpu'))
    model.eval()
    return model_type

def predict(img):
    img = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
    probs = torch.softmax(logits, dim=1)
    pred = labels[probs.argmax()][0]
    return pred

# gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Dropdown(["coatnet", "densenet161", "efficientnetv2", "mobilevit", "resnet", "swintransformer"], label="Model", info="Select the finetuned model to use."),
#         gr.Image(type="pil"),
#     ],
#     outputs=gr.Label(num_top_classes=1),
#     examples=[examples],
# ).launch()


with gr.Blocks() as demo:

    # model_name = gr.Dropdown(["coatnet", "densenet161", "efficientnetv2", "mobilevit", "resnet", "swintransformer"], label="Model", info="Select the finetuned model to use."),
    # loaded_model = gr.Textbox(value="", label="Loaded Model")
    # btn = gr.Button(value="Load Model")
    # btn.click(load_model, inputs=model_name, outputs=[loaded_model])

    with gr.Row():
        im = gr.Image()
        txt_output = gr.Textbox(num_top_classes=1)

    btn = gr.Button(value="Predict Car Model")
    btn.click(predict, inputs=[im], outputs=[txt_output])

    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=example_images,
        inputs=im,
        outputs=txt_output,
    )

if __name__ == "__main__":
    demo.launch()