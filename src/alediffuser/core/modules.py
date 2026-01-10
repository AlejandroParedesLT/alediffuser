import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_dim:int, output_dim:int,device:torch.device,dtype:torch.dtype):
        super().__init__()
        self.fn=nn.Sequential(
            nn.Linear(input_dim,output_dim,device=device,dtype=dtype),
            nn.ReLU()
        )
    def forward(self,x):
        return self.fn(x)    

class PositionalEmbedding(nn.Module):
    def __init__(
        self, 
        T:int, # Timesteps
        output_dim:int, # dimension on output
        device,
        dtype,
        **kwargs):
        super().__init__()
        self.output_dim=output_dim
        # Create an array of T positions [[1,1,1,1,1]] 
        position=torch.arange(T,device=device,dtype=dtype).unsqueeze(1)
        # PE(pos,2i)=sin(pos/1000^(2i/d_model))
        div_term=torch.exp(torch.arange(0,output_dim,2,dtype=dtype,device=device)*(-math.log(1000.0)/output_dim))
        # PE(pos,2i)=sin(pos/1000^(2i/d_model))
        pe=torch.zeros(T,output_dim,device=device,dtype=dtype)
        pe[:, 0::2]=torch.sin(position*div_term) # Apply positional encoding to even
        pe[:, 1::2]=torch.cos(position*div_term) # positional embedding to odd
        self.register_buffer('pe',pe)
        
    def forward(
            self,
            x:torch.Tensor # Batch Integer timestep
        ):
        return self.pe[x].reshape(x.shape[0],self.output_dim)
    

class TimeEmbedding(nn.Module):
    def __init__(
        self, 
        T:int, 
        hidden_dim,
        output_dim,
        device,
        dtype,
        **kwargs):
        super().__init__()
        self.positionalEmbedding=PositionalEmbedding(T=T,output_dim=hidden_dim,device=device,dtype=dtype)
        self.seq=nn.Sequential(
            nn.Linear(hidden_dim,output_dim,device=device,dtype=dtype),
            nn.SiLU(),
            nn.Linear(output_dim,output_dim,device=device,dtype=dtype)
        )

    def forward(self,x):
        pos_emb=self.positionalEmbedding(x)
        return self.seq(pos_emb)
        


# class multihead_self_attention(nn.Module):
#     # Where d_model is the embedding size
#     def __init__(
#             self, 
#             d_model,
#             num_heads,
#             input_dim
#             ):
#         super().__init__()
#         assert d_model%num_heads==0, 'Incorrect dimensions'
#         self.d_model=d_model # [batch, seq_size, embedding_size=d_model]
#         self.d_head=d_model//num_heads
#         self.W_q=nn.Linear(d_model,d_model)
#         self.W_k=nn.Linear(d_model,d_model)
#         self.W_v=nn.Linear(d_model,d_model)
        
#         self.W_o=nn.Linear(d_model,d_model)
#         self.norm=nn.LayerNorm([input_dim])
#         self.mlp=nn.Sequential(
#             nn.LayerNorm([input_dim]),
#             nn.Linear(input_dim,input_dim),
#             nn.GELU(),
#             nn.Linear(input_dim,input_dim)
#         )
#     def forward(self,x,t,condition,mask):
#         batch,input_dim,h,w=x.shape
#         # proj_q=self.
#         # TODO: (Alejandro) complete the forward pass



def Normalize(in_channels, num_groups=32,device=None,dtype=None):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True,device=device,dtype=dtype)


class SpatialSelfAttention(nn.Module):
    def __init__(self, n_channel,device,dtype):
        super().__init__()
        self.n_channel=n_channel
        self.norm=Normalize(in_channels=n_channel,device=device,dtype=dtype)
        self.q=nn.Conv2d(
            in_channels=n_channel,
            out_channels=n_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            device=device,
            dtype=dtype
        )
        self.k=nn.Conv2d(
            in_channels=n_channel,
            out_channels=n_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            device=device,
            dtype=dtype
        )
        self.v=nn.Conv2d(
            in_channels=n_channel,
            out_channels=n_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            device=device,
            dtype=dtype
        )
        self.proj_out=nn.Conv2d(
            in_channels=n_channel,
            out_channels=n_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            device=device,
            dtype=dtype
        )
    def forward(self,x:torch.Tensor):
        h_=x
        h_=self.norm(h_)
        q=self.q(h_)
        k=self.k(h_)
        v=self.v(h_)
        
        # Computes attention
        b,c,h,w=q.shape
        q=q.reshape(b,c,h*w)
        q=q.permute(0,2,1) # this is doing b,hw,c
        k=k.reshape(b,c,h*w)
        w_=torch.bmm(q,k) # [b, hw, hw]
        w_=w_*(int(c)**(-0.5))
        w_=nn.functional.softmax(w_,dim=2)
        # attend to values
        v=v.reshape(b,c,h*w)
        w_=w_.permute(0,2,1) # b,hw,hw (first hw of k, second of q)
        h_=torch.bmm(v,w_)
        h_=h_.reshape(b,c,h,w)
        h_=self.proj_out(h_)
        return x+h_



class DownSample(nn.Module):
    def __init__(self, in_channels,with_conv:bool,device,dtype):
        super().__init__()
        self.in_channels=in_channels
        self.with_conv=with_conv
        if self.with_conv:
            self.conv=nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
                device=device,
                dtype=dtype
            )
    def forward(self, x):
        if self.with_conv:
            pad=(0,1,0,1)
            x=nn.functional.pad(x,pad,mode='constant',value=0)
            x=self.conv(x)
        else:
            x=nn.functional.avg_pool2d(x,kernel_size=2,stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels,with_conv,device,dtype):
        super().__init__()
        self.with_conv=with_conv
        if self.with_conv:
            self.conv=nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                device=device,
                dtype=dtype
            )
    def forward(self, x):
        x=nn.functional.interpolate(x,scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x=self.conv(x)
        return x

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(
            self, 
            *, 
            in_channels, 
            out_channels=None, 
            conv_shortcut=False,
            dropout,
            temb_channels=512,
            device,
            dtype
        ):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=in_channels if out_channels is None else out_channels
        self.use_conv_shortcut=conv_shortcut
        
        # stage 1
        self.norm1=Normalize(in_channels=in_channels,device=device,dtype=dtype)
        self.conv1=nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype
            )
        if temb_channels>0:
            self.temb_proj=nn.Linear(
                temb_channels,
                out_channels,
                device=device,
                dtype=dtype)
            
        self.norm2=Normalize(out_channels,device=device,dtype=dtype)
        self.dropout=nn.Dropout(dropout)
        self.conv2=nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype)
        
        if self.in_channels!=self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut=nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=device,
                    dtype=dtype)
            else:
                self.nin_shortcut=nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    device=device,
                    dtype=dtype)
            
    def forward(self, x, temb):
        h=x
        h=self.norm1(h)
        h=nonlinearity(h)
        h=self.conv1(h)
        if temb is not None:
            h=h+self.temb_proj(nonlinearity(temb))[:,:,None,None]
        h=self.norm2(h)
        h=nonlinearity(h)
        h=self.dropout(h)
        h=self.conv2(h)
        
        if self.in_channels!=self.out_channels:
            if self.use_conv_shortcut:
                x=self.conv_shortcut(x)
            else:
                x=self.nin_shortcut(x)
            
        return x+h
            
    
class SequenceWithTimeEmbedding(nn.Module):
    def __init__(self,blocks):
        super().__init__()
        self.models=nn.ModuleList(blocks)
    
    def forward(self, x, t, cond, mask):
        for model in self.models:
            x=model(x,t,cond,mask)
        return x
    
