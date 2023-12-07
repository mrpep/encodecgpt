import numpy as np
import torch
from tqdm import tqdm

def roll_along(arr, shifts, dim):
    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim], device=arr.device).reshape(shape)
    indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
    
    return torch.gather(arr, dim, indices)

def generate_interp(prompts, model, buffer_size=300, temperature=0.7, generation_steps=500):
    if prompts.ndim==1:
        prompts = np.tile(prompts[np.newaxis,:],(generation_steps,1))
    if not isinstance(temperature, np.ndarray):
        temperature = temperature*np.ones((len(prompts),))
    with torch.no_grad():
        prompts = torch.from_numpy(prompts).to(model.device, dtype=model.dtype)
        gpt_in = prompts[0].unsqueeze(0).unsqueeze(0)
        generation = []
        for i,p in tqdm(enumerate(prompts)):
            gpt_in[:,0] = p
            outs = model.gpt(inputs_embeds=gpt_in)
            preds = model.classification_head(outs['last_hidden_state'][:,-1,:])
            preds = preds.view(preds.shape[0],model.num_q,model.num_codes+1)
            sampled_idxs = torch.cat([torch.multinomial(torch.nn.functional.softmax(preds[0,q,:]/temperature[i]),1) for q in range(model.num_q)])
            generation.append(sampled_idxs)
            if i<buffer_size:
                in_idxs = torch.arange(model.num_q, device=model.device)*(model.num_codes + 1) + sampled_idxs
                gpt_in = torch.cat([gpt_in, model.vocab_embed(in_idxs).sum(axis=0).unsqueeze(0).unsqueeze(0)],axis=1)
            else:
                generation_ = torch.stack(generation)[-buffer_size:]
                for k in range(model.num_q-1):
                    generation_[k,k+1:] = 0
                in_idxs = (torch.arange(model.num_q, device=model.device)[None,:])*(model.num_codes + 1) + generation_
                gpt_seq = model.vocab_embed(in_idxs).sum(axis=1).unsqueeze(0)
                gpt_in = torch.cat([gpt_in[:,0].unsqueeze(1), gpt_seq],axis=1)
                    
        generation = torch.stack(generation)
        generation = roll_along(generation,-torch.arange(0,8,device=generation.device),0)
        
        audio = model.encodec_model.decode([(torch.maximum(generation-1, torch.tensor(0, device=model.device))[:-model.num_q].T.unsqueeze(0),None)])
        audio = audio[0].cpu().detach().numpy()
        return audio, generation