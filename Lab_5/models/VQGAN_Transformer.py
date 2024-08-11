import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_q ,codebook_indices, _ = self.vqgan.encode(x)
        codebook_indices = codebook_indices.view(z_q.shape[0], -1)
        return codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda x: 1 - x
        elif mode == "cosine":
            return lambda x: np.cos(np.pi * x/2)
        elif mode == "square":
            return lambda x: 1 - x**2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        '''
        During training, we sample a subset of tokens and replace them with a special [MASK] token.
        we first sample a ratio from 0 to 1, then uniformly select [r*N ] tokens in Y to place masks, where N is the length. 
        '''
        z_indices = self.encode_to_z(x)
        mask_token=math.ceil(self.gamma(np.random.uniform()) * z_indices.shape[1])
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(
            dim=1,
            index=torch.randperm(z_indices.shape[1], device=z_indices.device)[:mask_token].unsqueeze(0).expand(z_indices.shape[0], -1),
            value=True
        )
    
        masked_indices = (~mask) * z_indices + mask * torch.full_like(z_indices, self.mask_token_id)
        logits = self.transformer(masked_indices)

        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, ratio, z_indices, mask, mask_num,gamma_func):
        #將mask的token值設為0'
        mask_indices = torch.where(mask, torch.full_like(z_indices, self.mask_token_id), z_indices)
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(self.transformer(mask_indices), dim=-1) 

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = logits.max(dim=-1)


        #predicted probabilities add temperature annealing gumbel noise as confidence G=−log(−log(U))
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob) + 1e-20) + 1e-20)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence[~mask] = float('inf')
        #define how much the iteration remain predicted tokens by mask scheduling
        num_keep_tokens =math.ceil(self.gamma_func(mode=gamma_func)(ratio) * mask_num)
        if num_keep_tokens <= 0:
            #if num_keep_tokens is 0, then all tokens are not be masked
            mask_bc = torch.zeros_like(mask, dtype=torch.bool)
        else:
            last_value = confidence.topk(num_keep_tokens, dim=-1, largest=False).values[0, -1]
            mask_bc = confidence <= last_value
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        z_indices_predict=torch.where(mask, z_indices_predict, z_indices)
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}