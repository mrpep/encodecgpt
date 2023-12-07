from encodecmae import load_model
from tqdm import tqdm
import numpy as np
from pathlib import Path
from loguru import logger
import gin
import random
import pandas as pd

def segment_dataset(state, segment_size=4, min_segments=1, max_segments=None):
    if ('dataset_metadata' in state) and ('segmented' in state) and state['segmented']:
        pass
    else:
        new_rows = []
        for idx, row in tqdm(state.dataset_metadata.iterrows()):
            duration = row['duration']
            if duration < segment_size:
                new_rows.append(row)
            else:
                n_segments = max(min_segments,duration//segment_size)
                if max_segments is not None:
                    n_segments = min(n_segments, max_segments)
                frames = row['frames']
                for i in range(int(n_segments)):
                    r = row.to_dict()
                    s = random.randint(0,frames-int(segment_size*row['sr']))
                    e = s+int(segment_size*row['sr'])
                    r['start'] = s
                    r['end'] = e
                    r['duration'] = segment_size
                    r['frames'] = e - s
                    new_rows.append(r)
        state.dataset_metadata = pd.DataFrame(new_rows)
        state.segmented = True
    return state

def make_prompt(state, model_fn=None, prompt_fn=None, output_path='prompts',cache=True):
    prompt_filenames = []
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    for idx, row in tqdm(state.dataset_metadata.iterrows()):
        if 'start' in row:
            out_filename = Path(output_path,str(row['rel_path']).replace('/','_').replace('.wav','')) / Path('{}-{}.npy'.format(row['start'], row['end']))
        else:
            out_filename = Path(output_path,str(row['rel_path']).replace('/','_').replace('.wav','.npy'))
        if not out_filename.exists():
            if not out_filename.parent.exists():
                out_filename.parent.mkdir(parents=True)
            if 'prompt_model' not in state:
                state.prompt_model = model_fn()
                logger.warning('If the prompt model uses gin, following tasks will fail. Rerun the script')
            if 'start' in row:
                s = row['start']
            else:
                s = None
            if 'end' in row:
                e = row['end']
            else:
                e = None
            state, prompt = prompt_fn(state,row['filename'],start=s,end=e)
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

def encodecmae_prompt(state, filename, start=None, end=None):
    features = state.prompt_model.extract_features_from_file(filename, start=start, end=end)
    return state, features[0].mean(axis=0)

def use_partition(df):
    partitions = {}
    for k in df['partition'].unique():
        partitions[k] = df.loc[df['partition'] == k]
    return partitions

def dynamic_pad_batch(batch):
    batch = {k: [b[k] for b in batch] for k in batch[0].keys()}
    from IPython import embed; embed()