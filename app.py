from fastai.vision.all import *
import gradio as gr

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Label()).launch(share=True)

demo.launch()