from encodec import EncodecModel
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from transformers import GPT2Model

def roll_along(arr, shifts, dim):
    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim], device=arr.device).reshape(shape)
    indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
    
    return torch.gather(arr, dim, indices)

def apply_delay_pattern(codes):
    codes = torch.nn.functional.pad(codes+1,(0,codes.shape[1]-1))
    codes = roll_along(codes, torch.arange(0,codes.shape[1], device=codes.device)[None,:].tile((codes.shape[0],1)), 2)
    return codes

def unapply_delay_pattern(codes):
    codes = roll_along(codes, -torch.arange(0,codes.shape[1], device=codes.device)[None,:].tile((codes.shape[0],1)), 2)
    codes = codes[:,:,:-codes.shape[1]]
    return codes

class EnCodecGPT(pl.LightningModule):
    def __init__(self, num_q=8, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.num_q = num_q
        self.num_codes = 1024
        self.vocab_size = self.num_q * (self.num_codes + 1)
        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.vocab_embed = torch.nn.Embedding(self.vocab_size, self.gpt.embed_dim)
        self.classification_head = torch.nn.Linear(self.gpt.embed_dim, self.vocab_size)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, x, prompt=None):
        #Encodec tokens:
        with torch.no_grad():
            codes = self.encodec_model.encode(x)[0][0][:,:self.num_q]
        #Delay pattern:
        codes = apply_delay_pattern(codes)
        #Offset each quantizer by 1025 and pass through LUT:
        input_vocab_embed = torch.arange(self.num_q, device=x.device)[None,:,None]*(self.num_codes + 1) + codes
        gpt_in = self.vocab_embed(input_vocab_embed).sum(axis=1)
        #Prepend prompt:
        if prompt is not None:
            gpt_in = torch.cat([prompt[:,None,:],gpt_in],axis=1)
        #Pass through GPT:
        gpt_out = self.gpt(inputs_embeds = gpt_in)
        #Make classification:
        preds = self.classification_head(gpt_out['last_hidden_state'])
        preds = preds.view(preds.shape[0],preds.shape[1],self.num_q,self.num_codes+1)
        return preds, codes

    def training_step(self,x, batch_idx):
        wav = x['wav'].unsqueeze(1)
        preds, targets = self(wav, x['prompt'])
        preds = preds.transpose(1,3)[:,:,:,:-1]
        loss = torch.nn.functional.cross_entropy(preds, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self,x, batch_idx):
        wav = x['wav'].unsqueeze(1)
        preds, targets = self(wav, x['prompt'])
        preds = preds.transpose(1,3)[:,:,:,:-1]
        loss = torch.nn.functional.cross_entropy(preds, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        opt = self.optimizer(self.trainer.model.parameters())

        if self.lr_scheduler is not None:
            if self.lr_scheduler.__name__ == 'SequentialLR':
                binds = gin.get_bindings('torch.optim.lr_scheduler.SequentialLR')
                lr_scheduler = self.lr_scheduler(opt, schedulers=[s(opt) for s in binds['schedulers']])
            else:
                lr_scheduler = self.lr_scheduler(opt) if self.lr_scheduler is not None else None
        else:
            lr_scheduler = None
        del self.optimizer
        del self.lr_scheduler
        opt_config = {'optimizer': opt}
        if lr_scheduler is not None:
            opt_config['lr_scheduler'] = {'scheduler': lr_scheduler,
                                          'interval': 'step',
                                          'frequency': 1}
        return opt_config

