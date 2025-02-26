import os
import gradio as gr
from models import SaigonBeer_Recognition

cfg = {"Model": {
              "classifier": r".\weights\classification\best.pt",
              "detector": r".\weights\detection\best.pt",
}}

sample_images = [
    "examples/samples/saigon_export.jpg",
    "examples/samples/saigon_chill.jpg",
    "examples/samples/saigon_large.jpg",
    "examples/samples/saigon_special.jpg",
    "examples/samples/saigon_gold.jpg",
    "examples/samples/others.jpg"
]

beer_recognition = SaigonBeer_Recognition(cfg)
def predict(image):
    saved_path = r"E:\CODE-learning\interview\Momo\Momo_interview\beer_recognition\examples\results"
    result, annotated_image =beer_recognition.forward(image, save_path=saved_path)          
    return annotated_image

title = "Ứng dụng nhận diện bia🍺"
description = "Tải lên một ảnh của bạn để mô hình nhận diện bia hoặc thử các ảnh mẫu bên dưới."

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Ảnh cần nhận diện"),
    outputs=gr.Image(type="pil", label="Kết quả dự đoán"),
    examples=sample_images,
    title=title,
    description=description,
    theme="default",
)

interface.launch()
