import gradio as gr
from encodecmae import load_model as load_encodecmae
import gin
import joblib
from ginpipe.core import setup_gin
from encodecmae.tasks import fit_model
import torch
import librosa
from gen_utils import generate_interp
import numpy as np

models_metadata = {'Librispeech': {'state': 'experiments/encodecgpt2/librilight-24k-encodecmaebase/state.pkl',
                                  'checkpoint': 'experiments/encodecgpt2/librilight-24k-encodecmaebase/checkpoints/last.ckpt'},
                   'NSynth': {'state': 'experiments/encodecgpt2/nsynth-24k-encodecmaebase/state.pkl',
                              'checkpoint': 'experiments/encodecgpt2/nsynth-24k-encodecmaebase/checkpoints/last2.ckpt'}}

gin.enter_interactive_mode()
def load_from_state(file_path):
    gin.clear_config()
    state = joblib.load(file_path)
    abs_flags = state['flags']
    state_ = setup_gin(abs_flags, save_config=False)
    model = gin.get_bindings(fit_model)['model_cls']()
    return model

def load_model(state, model):
    gin.clear_config()
    state['encodecmae'] = load_encodecmae('base', device='cuda:1')
    state['model_name'] = model
    model_paths = models_metadata[model]
    model = load_from_state(model_paths['state'])
    model = model.to('cuda:1')
    model.load_state_dict(torch.load(model_paths['checkpoint'])['state_dict'])
    state['encodecmae'].visible_encoder.compile=False
    state['model'] = model
    return state

def update_model(state, model):
    if model != state['model_name']:
        state['model_name'] = model
        model_paths = models_metadata[model]
        model = load_from_state(model_paths['state'])
        model = model.to('cuda:1')
        model.load_state_dict(torch.load(model_paths['checkpoint'])['state_dict'])
        state['model'] = model
    return state
        
def generate(state, temperature, prompt, duration):
    sr, prompt = prompt
    prompt = prompt/(2**15-1)
    target_sr=24000
    if sr != target_sr:
        prompt = librosa.resample(prompt, orig_sr=sr, target_sr=target_sr)
    prompt_vector = state['encodecmae'].extract_features_from_array(prompt[np.newaxis,:])[0].mean(axis=0)
    print(state['model_name'])
    audio, _ = generate_interp(prompt_vector, state['model'], temperature=temperature, generation_steps=int(duration*75), buffer_size=200)

    return (target_sr, audio[0]), state

with gr.Blocks() as demo:
    session_state = gr.State({})
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(choices=['NSynth', 'Librispeech'], value='NSynth',label='Model')
            duration = gr.Slider(minimum=1, maximum=15,value=3,label='Duration')
            temperature = gr.Slider(minimum=0.01, maximum=2,value=0.6,label='Temperature')
        with gr.Column():
            prompt = gr.Audio()
            submit_btn = gr.Button('Generate')
    with gr.Row():
        result = gr.Audio()
    
    demo.load(load_model, inputs=[session_state, model], outputs=[session_state])
    submit_btn.click(generate, inputs=[session_state, temperature, prompt, duration], outputs=[result, session_state])
    model.change(update_model,inputs=[session_state, model], outputs=[session_state])
    
    
if __name__ == "__main__":
    demo.launch(show_api=False)