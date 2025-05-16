"""
Basic implementation of a Denoising Diffusion Probabilistic Model (DDPM)
"""

import torch
from torch import Tensor
import torch.nn as nn
import math
from tqdm import tqdm
import pytorch_lightning as pl
from copy import deepcopy
from models.unet import Unet


#1. Implements useful functions.

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from 
    
        https://github.com/Lightning-AI/pytorch-lightning/issues/10914

    """
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)



class DDPM(pl.LightningModule):
    
    def __init__(self, image_size, in_channels, time_embedding_dim = 256, timesteps = 1000, base_dim = 32, dim_mults = [1,2,4,8], lr = 5*1e-4, total_steps = None):
        """
        Inputs:
            image_size:
            in_channels:
            time_embedding_dim:
            timesteps:
            base_dim
            dim_mults
        """
        super().__init__()

        self.save_hyperparameters()
        self.timesteps = torch.tensor([timesteps], dtype = torch.long)
        self.in_channels = in_channels
        self.image_size = image_size
        self.lr = lr
        self.total_steps = total_steps


        #. Defines noise parameters.

        betas = self.cosine_variance_schedule(self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim = -1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

        self.model = Unet(self.timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults)
        self.model_ema = EMA(self.model, decay= 0.99)

    def forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape, "Sample x and noise must have same shape."

        sample = self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise
        
        return sample

    def forward(self, x, noise):

        t =  torch.randint(0, self.timesteps, (x.shape[0], ), device= self.device)
        #t = torch.randint_like(x)


        x_t = self.forward_diffusion(x, t, noise)

        pred_noise = self.model(x_t, t)

        return pred_noise

    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True):
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(self.device)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t)
            t=torch.tensor([i for _ in range(n_samples)]).to(self.device)

            if clipped_reverse_diffusion:
                x_t=self.reverse_diffusion_with_clip(x_t,t,noise).to(self.device)
            else:
                x_t=self.reverse_diffusion(x_t,t,noise).to(self.device)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t

    def cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps.item(),steps=timesteps.item()+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
    
    @torch.no_grad()
    def reverse_diffusion(self,x_t,t,noise):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 

    @torch.no_grad()
    def reverse_diffusion_with_clip(self,x_t,t,noise): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),lr= self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,self.lr,total_steps= self.total_steps,pct_start=0.25,anneal_strategy='cos')

        return [optimizer], [scheduler]

    def loss_function(self, x_hat, x):
        mse_loss = nn.MSELoss()
        loss = mse_loss(x_hat, x)

        return loss

    def training_step(self, batch, batchidx):
        #a. Retrieves data. 
        images  = batch[0]
        noise = torch.randn_like(images) # Original: Without indexing.
        
        #b. Model prediction.

        pred = self(images, noise)

        #c. Calculates loss.

        loss = self.loss_function(pred, noise)
        self.log('loss', loss, prog_bar= True)

        return loss

    def validation_step(self, batch, batch_idx):
        return  

    def on_before_backward(self, loss: Tensor) -> None:
        self.model_ema.update(self.model)

    def gen_sample(self, N = 1):
        samples = self.sampling(n_samples= N)

        return samples
   
def create_ddpm(timesteps = 100, image_size = 400, in_channel = 1, base_dim = 16, dim_mults = [2, 4], total_steps_factor = 256):
    
    #a. Defines architecture.
    total_steps = total_steps_factor*timesteps
    model = DDPM(image_size= image_size, in_channels= in_channel, timesteps= timesteps, base_dim= base_dim, dim_mults= dim_mults, total_steps= total_steps)

    return model
