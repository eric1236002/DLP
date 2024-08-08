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
        _,codebook_indices, _ = self.vqgan.encode(x)
        codebook_indices = codebook_indices.view(codebook_indices.shape[0], -1)
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
            return lambda x: 0.5*(1 + math.cos(math.pi * x))
        elif mode == "square":
            return lambda x: 1 - x**2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        '''
        During training, we sample a subset of tokens and replace them with a special [MASK] token.
        '''
        z_indices = self.encode_to_z(x)
        z_indices = z_indices.view(x.size(0), -1)
        mask_ratio = self.gamma(torch.rand(1).item())
        mask = torch.rand(z_indices.size(0), z_indices.size(1), device=x.device) < mask_ratio
        mask_indices = torch.where(mask, z_indices, torch.tensor(self.mask_token_id, device=x.device))
        
        logits = self.transformer(mask_indices)  #transformer predict the probability of tokens

        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, ratio, z_indices, mask, mask_num):
        #將mask的token值設為0'
        z_indices[mask] = 0
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(self.transformer(z_indices), dim=-1) 

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = logits.max(dim=-1)


        #predicted probabilities add temperature annealing gumbel noise as confidence G=−log(−log(U))
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob) + 1e-20) + 1e-20)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        

        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence[~mask] = float('inf')
        #sort the confidence for the rank 
        sorted_confidence, sorted_indices = torch.sort(confidence, descending=True)
        #define how much the iteration remain predicted tokens by mask scheduling
        num_keep_tokens = int((1 - ratio) * mask_num)
        mask_bc = torch.ones_like(mask, dtype=torch.bool)
        mask_bc[sorted_indices[:, :num_keep_tokens]] = False
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        z_indices_predict[~mask] = z_indices[~mask]
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}