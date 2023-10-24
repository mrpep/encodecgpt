from encodecmae import load_model
from tqdm import tqdm
import numpy as np
from pathlib import Path
from loguru import logger

def make_prompt(state, model_fn=None, prompt_fn=None, output_path='prompts',cache=True):
    prompt_filenames = []
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for idx, row in tqdm(state.dataset_metadata.iterrows()):
        out_filename = Path(output_path,str(row['rel_path']).replace('/','_').replace('.wav','.npy'))
        if not out_filename.exists():
            if 'prompt_model' not in state:
                state.prompt_model = model_fn()
                logger.warning('If the prompt model uses gin, following tasks will fail. Rerun the script')
            state, prompt = prompt_fn(state,row['filename'])
            np.save(out_filename, prompt)
        prompt_filenames.append(out_filename)
    state.dataset_metadata['prompt'] = prompt_filenames
    if 'prompt_model' in state:
        del state['prompt_model']
    return state

def load_encodecmae(model='base', device='cuda:0'):
    if isinstance(device, list):
        device = 'cuda:{}'.format(device[0])
    elif isinstance(device, int):
        device = 'cuda:{}'.format(device)
    config_str = gin.config_str()
    gin.clear_config()
    model = load_model(model, device=device)
    gin.parse_config(config_str)
    return model

def encodecmae_prompt(state, filename):
    features = state.prompt_model.extract_features_from_file(filename)
    return state, features.mean(axis=0)

def use_partition(df):
    partitions = {}
    for k in df['partition'].unique():
        partitions[k] = df.loc[df['partition'] == k]
    return partitions