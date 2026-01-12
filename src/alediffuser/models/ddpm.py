import matplotlib.pyplot as plt
# from scipy.stats import norm
import torch.nn as nn
import torch

class DDPM(nn.Module):
    def __init__(
        self,
        T: int, # Timesteps
        p_cond:float,
        eps_model: nn.Module,
        device:str,
        dtype:torch.dtype
    ):
        super().__init__() # Initialize module
        self.T = T # Timesteps
        self.eps_model=eps_model.to(device)
        self.device=device
        self.dtype=dtype
        self.p_cond = torch.tensor([p_cond],dtype=dtype).to(device)
        # Schedules
        beta_schedule=torch.linspace(1e-4,0.02,T+1,device=device,dtype=dtype)
        alpha_t_schedule=1-beta_schedule
        bar_alpha_t_schedule=torch.cumprod(alpha_t_schedule,axis=0)
        sqrt_bar_alpha_t_schedule=torch.sqrt(bar_alpha_t_schedule)
        sqrt_minus_bar_alpha_t_schedule=torch.sqrt(1-bar_alpha_t_schedule)
        self.register_buffer("beta_schedule",beta_schedule)
        self.register_buffer("alpha_t_schedule",alpha_t_schedule)
        self.register_buffer("bar_alpha_t_schedule",bar_alpha_t_schedule)
        self.register_buffer("sqrt_bar_alpha_t_schedule",sqrt_bar_alpha_t_schedule)
        self.register_buffer("sqrt_minus_bar_alpha_t_schedule",sqrt_minus_bar_alpha_t_schedule)
        
        
    def forward(self,imgs:torch.Tensor,conds:torch.Tensor):
        # Random 
        t=torch.randint(low=1,high=self.T+1, size=(imgs.shape[0],),device=self.device)
        
        # get random noise to add it to the images
        noise=torch.randn_like(imgs,device=self.device,dtype=self.dtype)
        
        # get noise image as: sqrt(alpha_t_bar) * x0 + noise * sqrt(1-alpha_t_bar)
        batch_size,channels,width,height=imgs.shape
        noise_imgs=self.sqrt_bar_alpha_t_schedule[t].view((batch_size,1,1,1))*imgs\
            + self.sqrt_minus_bar_alpha_t_schedule[t].view((batch_size,1,1,1))*noise
            
        conds=conds.unsqueeze(1)
        mask=torch.rand_like(conds,dtype=torch.float32)>self.p_cond

        # TODO: Add the conditionals to the entire network

        pred_noise=self.eps_model(noise_imgs,t,conds,mask)
        # return self.criterion(pred_noise,noise)
        return pred_noise,noise
    
    def sample(self, n_samples, size):
        self.eval()
        with torch.no_grad():
            x_t = torch.randn(n_samples, *size, device=self.device, dtype=self.dtype)
            for t in range(self.T, 0,-1):
                t_tensor=torch.tensor([t],device=self.device).repeat(x_t.shape[0],1)
                pred_noise=self.eps_model(x_t, t_tensor,None,None)
                
                z=torch.randn_like(x_t,device=self.device,dtype=self.dtype) if t>0 else 0
                
                # x_(t-1) = 1 / sqrt(alpha_t) * (x_t - pred_noise * (1 - alpha_t) / sqrt(1 - alpha_t_bar)) + beta_t * eps
                x_t=1/torch.sqrt(self.alpha_t_schedule[t])*\
                    (x_t - pred_noise * (1-self.alpha_t_schedule[t])/self.sqrt_minus_bar_alpha_t_schedule[t]) +\
                        torch.sqrt(self.beta_schedule[t])*z
            return x_t
        
# class LatentDiffusion(DDPM):
#     def __init__(
#         self,
#         first_stage_config,
#         cond_stage_config,
#         num_timesteps_cond=None,
#         cond_stage_key="image",
#         cond_stage_trainable=False,
#         concat_mode=True,
#         cond_stage_forward=None,
#         conditioning_key=None,
#         scale_factor=1.0,
#         scale_by_std=False,

#     )