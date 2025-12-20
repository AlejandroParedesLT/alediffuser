import torch
import torch.nn as nn
import math
from alediffuser.core.modules import *

class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels, # 3 rgb
        out_channels, # 3 rgb
        T,
        steps=(1,2,4,8),
        hid_size=128,
        attention_step_indexes=[1,2,3],
        has_residuals=True,
        num_resolution_blocks=4,
        is_debug=False,
        resamp_with_conv=True,
        dropout=0.0,
        num_classes=None
        ):
        super().__init__()
        time_emb_dim=hid_size*4
        self.num_classes=num_classes
        self.num_resolution_blocks=num_resolution_blocks
        self.ch_mult=steps

        # In and time embedding
        self.time_embedding=TimeEmbedding(T,hid_size,time_emb_dim)

        # First convolution
        self.firs_conv=nn.Conv2d(
            in_channels=in_channels,
            out_channels=steps[0]*hid_size,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        if num_classes:
            self.cond_embedding = nn.Sequential(
                nn.Embedding(num_classes, hid_size),
                nn.Linear(hid_size, time_emb_dim),
                nn.ReLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
        
        # resnet blocks, downsample convolutions
        in_ch_mult = (1,)+tuple(self.ch_mult)
        self.down_sample=nn.ModuleList()
        prev_hid_size = self.ch_mult[0] * hid_size
        for (i_level, step) in enumerate(self.ch_mult):
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in=hid_size*in_ch_mult[i_level]
            block_out=hid_size*self.ch_mult[i_level]
            for block in range(num_resolution_blocks):
                res_block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=time_emb_dim,
                        dropout=dropout
                    )
                )
                block_in=block_out
                if i_level in attention_step_indexes:
                    attn_block.append(
                        SpatialSelfAttention(
                            n_channel=block_in
                        )
                    )
            downSampleBlock=nn.Module()
            downSampleBlock.res_block=res_block
            downSampleBlock.attention=attn_block
            if i_level!=len(self.ch_mult) - 1:
                downSampleBlock.downsample=DownSample(
                    in_channels=block_in,
                    with_conv=resamp_with_conv
                    )
            self.down_sample.append(downSampleBlock)
        
        # Middle-bottleneck, latent space
        self.mid=nn.Module()
        self.mid.block1=ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_in,
                        temb_channels=time_emb_dim,
                        dropout=dropout
                        )
        self.mid.atte1=SpatialSelfAttention(
            block_in
        )
        self.mid.block2=ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_in,
                        temb_channels=time_emb_dim,
                        dropout=dropout
                        )


        # Upsampling
        self.up_sample=nn.ModuleList()
        for i_level in reversed(range(len(self.ch_mult))):
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = hid_size* self.ch_mult[i_level]
            skip_in = hid_size* self.ch_mult[i_level]
            for i_block in range(num_resolution_blocks+1):
                if i_block == num_resolution_blocks:
                    skip_in = hid_size*in_ch_mult[i_level]
                res_block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=time_emb_dim,
                        dropout=dropout
                    )
                )
                block_in = block_out
                if i_level in attention_step_indexes:
                    attn_block.append(
                        SpatialSelfAttention(
                            n_channel=block_out
                        )
                    )
            upSampleBlock=nn.Module()
            upSampleBlock.res_block=res_block
            upSampleBlock.attention=attn_block
            if i_level!=0:
                upSampleBlock.upsample=Upsample(
                    in_channels=block_in,
                    with_conv=resamp_with_conv
                )
            self.up_sample.insert(0,upSampleBlock)
            # block_in = block_out

        self.norm_out=Normalize(block_in)
        self.conv_out=nn.Conv2d(
            block_in,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    def forward(self, x, t, cond, mask):
        time_emb=self.time_embedding(t)
        if self.num_classes:
            cond_emb=self.cond_embedding(cond)
        
        # Downsampling
        hs=[self.firs_conv(x)]
        for i_level in range(len(self.ch_mult)):
            for i_block in range(self.num_resolution_blocks):
                h=self.down_sample[i_level].res_block[i_block](hs[-1],time_emb)
                if len(self.down_sample[i_level].attention) > 0:
                    h=self.down_sample[i_level].attention[i_block](h)
                hs.append(h)
            
            if i_level != len(self.ch_mult) - 1:
                hs.append(self.down_sample[i_level].downsample(hs[-1]))
        
        # Bottleneck
        h=hs[-1]
        h=self.mid.block1(h,time_emb)
        h=self.mid.atte1(h)
        h=self.mid.block1(h,time_emb)

        # Upssampling
        for i_level in reversed(range(len(self.ch_mult))):
            for i_block in range(self.num_resolution_blocks+1):
                h=self.up_sample[i_level].res_block[i_block](
                    torch.cat([h,hs.pop()],dim=1),time_emb
                )
                if len(self.up_sample[i_level].attention)>0:
                    h=self.up_sample[i_level].attention[i_block](h)
            if i_level != 0:
                h = self.up_sample[i_level].upsample(h)

            
        # End
        h=self.norm_out(h)
        h=nonlinearity(h)
        h=self.conv_out(h)
        return h
    

class Encoder(nn.Module):
    def __init__(
            self,
            *,
            ch, # Hidden dimension size
            ch_mult=(1,2,4,8),
            num_res_blocks,
            attn_resolutions,
            dropout,
            resamp_with_conv,
            in_channels,
            resolution,
            z_channels,
            double_z=True,
        ):
        super().__init__()
        self.ch=ch
        self.time_embedding_ch=0
        self.ch_mult=ch_mult,
        self.num_resolutions=len(ch_mult)
        self.num_res_blocks=num_res_blocks,
        self.attn_resolutions=attn_resolutions,
        self.dropout=dropout,
        self.resamp_with_conv=resamp_with_conv,
        self.in_channels=in_channels,
        self.resolution=resolution,
        self.z_channels=z_channels,
        self.double_z=double_z

        # Downsampling path
        self.conv_in=nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        curr_res=resolution
        in_ch_mult=(1,)+tuple(ch_mult)
        self.in_ch_mult=in_ch_mult
        self.down_block=nn.ModuleList()
        for i_level in range(self.num_resolutions):
            resnet_block=nn.ModuleList()
            attn_block=nn.ModuleList()
            block_in=ch*in_ch_mult[i_level]
            block_out=ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.time_embedding_ch,
                        dropout=dropout
                    )
                )
                block_in=block_out
                if curr_res in attn_resolutions:
                    attn_block.append(
                        SpatialSelfAttention(
                            n_channel=block_out
                        )
                    )
            down_block=nn.Module()
            down_block.resnet_block=resnet_block
            down_block.attn_block=attn_block
            if i_level!= self.num_resolutions-1:
                down_block.downsample=DownSample(block_in,resamp_with_conv)
                curr_res=curr_res//2
            self.down_block=down_block
        
        # middle block
        self.mid_block=nn.Module()
        self.mid_block.block1=ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.time_embedding_ch,
            dropout=dropout
        )
        self.mid_block.attn1=SpatialSelfAttention(
            block_in
        )
        self.mid_block.block2=ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.time_embedding_ch,
            dropout=dropout
        )

        # end block
        self.norm_out = Normalize(block_in)

        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=2*z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        temb=None
        hs=[self.conv_in(x)]
        
        # downsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h=self.down_block[i_level].resnet_block[i_block](hs[-1],temb)
                if len(self.down_block[i_level].attn_block)>0:
                    h=self.down_block[i_level].attn_block[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down_block[i_level].downsample(hs[-1]))
            
        # middle
        h=hs[-1]
        h=self.mid_block.block1(h,temb)
        h=self.mid_block.attn1(h,temb)
        h=self.mid_block.block2(h,temb)

        # end
        h=self.norm_out(h)
        # Simplified SGWilu
        h=nonlinearity(h)
        h=self.conv_out(h)
        return h
    
import numpy as np
class Decoder(nn.Module):
    def __init__(
            self,
            *,
            ch,
            out_ch,
            ch_mult=(1,2,4,8),
            num_res_blocks,
            attn_resolutions,
            dropout,
            resamp_with_conv,
            in_channels,
            resolution,
            z_channels,
            double_z=True,
            give_pre_end=False, 
            tanh_out=False,
        ):
        super().__init__()
        self.ch=ch
        self.time_embedding_ch=0
        self.ch_mult=ch_mult,
        self.num_resolutions=len(ch_mult)
        self.num_res_blocks=num_res_blocks,
        self.attn_resolutions=attn_resolutions,
        self.dropout=dropout,
        self.resamp_with_conv=resamp_with_conv,
        self.in_channels=in_channels,
        self.resolution=resolution,
        self.z_channels=z_channels,
        self.double_z=double_z
        self.give_pre_end=give_pre_end
        self.tanh_out=tanh_out

        block_in=ch*ch_mult[self.num_resolutions-1]
        curr_res=resolution//2**(self.num_resolutions)
        self.z_shape=(1,z_channels,curr_res,curr_res)

        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv2d(
            in_channels=z_channels,
            out_channels=block_in,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # middle
        self.mid_block=nn.Module()
        self.mid_block.block1=ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.time_embedding_ch
        )
        self.mid_block.attn1=SpatialSelfAttention(
            block_in
        )
        self.mid_block.block2=ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.time_embedding_ch,
            dropout=dropout
        )

        self.up_block=nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            resnet_block=nn.ModuleList()
            attn_block=nn.ModuleList()
            block_out=ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                resnet_block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.time_embedding_ch,
                        dropout=dropout
                    )
                )
                block_in=block_out
                if curr_res in attn_resolutions:
                    attn_block.append(
                        SpatialSelfAttention(block_in)
                    )
            up_block=nn.Module()
            up_block.resnet_block=resnet_block
            up_block.attn_block=attn_block
            if i_level!=0:
                up_block.upsample=Upsample(in_channels=block_in,with_conv=resamp_with_conv)
                curr_res=curr_res*2
            self.up_block.insert(0,up_block)
        #end
        self.norm_out=Normalize(block_in)
        self.conv_out=nn.Conv2d(
            in_channels=block_in,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self,z):
        self.last_z_shape=z.shape

        temb=None

        # z to block_in (from latent space to up convolution)
        h=self.conv_in(z)

        # middle
        h=self.mid_block.block1(h,temb)
        h=self.mid_block.attn1(h,temb)
        h=self.mid_block.attn2(h,temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h=self.up_block[i_level].resnet_block[i_block](h,temb)
                if len(self.up_block[i_level].attn_block)>0:
                    h=self.up_block[i_level].attn[i_block](h)
            if i_level!=0:
                h=self.up_block[i_level].upsample(h)
            
        # end
        if self.give_pre_end:
            return h
    
        h=self.norm_out(h)
        h=nonlinearity(h)
        h=self.conv_out(h)
        if self.tanh_out:
            h=torch.tanh(h)
        return h