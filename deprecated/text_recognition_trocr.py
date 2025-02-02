from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import gradio as gr

def predict(image, processor, model):
    #predict the image using microsoft/trocr-large-handwritten model loaded earlier
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


title = "Handwritten Recognition app!" 

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten') 

image = Image.open("data/AS Z AG 12-1953_01_24_001.jpg").convert("RGB")

text = predict(image, processor, model)

print(text)

#interface = gr.Interface(fn=predict, inputs=["text",gr.Sketchpad(type="pil",shape=(500, 500)),gr.Image(type="pil")], outputs="text", title=title ) 

#interface.launch(server_name="0.0.0.0", server_port=8080) 


