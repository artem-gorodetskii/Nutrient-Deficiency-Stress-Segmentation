# -*- coding: utf-8 -*-
# @Author: Artem Gorodetskii
# @Created Time: 3/22/2022 4:45 PM

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, norm_type='GN'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.selu = nn.SELU()

        if norm_type == 'BN':
          self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'GN':
          self.norm = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.selu(x)
        out = self.norm(x)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, norm_type='GN', do_p=0.1):
        super(EncoderBlock, self).__init__()

        self.conv_block_1 = ConvBlock(in_channels, out_channels, norm_type=norm_type)
        self.dropout_1 = nn.Dropout2d(do_p)
        self.conv_block_2 = ConvBlock(out_channels, out_channels, norm_type=norm_type)
        self.dropout_2 = nn.Dropout2d(do_p)
        self.conv_block_3 = ConvBlock(out_channels, out_channels, norm_type=norm_type)

        if norm_type == 'BN':
          self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'GN':
          self.norm = nn.GroupNorm(32, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_3 = nn.Dropout(0.3)

    def forward(self, x):

        res_x = self.conv_block_1(x)
        x = self.dropout_1(res_x)
        x = self.conv_block_2(x)
        x = self.dropout_2(x)
        x = self.conv_block_3(x)
        x = torch.add(x, res_x)
        out_decoder = self.norm(x)
        x = self.pool(out_decoder)
        out = self.dropout_3(x)

        return out_decoder, out

class CodingBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, norm_type='GN', do_p=0.1):
        super(CodingBlock, self).__init__()

        self.conv_block_1 = ConvBlock(in_channels, out_channels, norm_type=norm_type)
        self.dropout_1 = nn.Dropout2d(do_p)
        self.conv_block_2 = ConvBlock(out_channels, out_channels, norm_type=norm_type)
        self.dropout_2 = nn.Dropout2d(do_p)
        self.conv_block_3 = ConvBlock(out_channels, out_channels, norm_type=norm_type)
        
        if norm_type == 'BN':
          self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'GN':
          self.norm = nn.GroupNorm(32, out_channels)

    def forward(self, x):

        res_x = self.conv_block_1(x)
        x = self.dropout_1(res_x)
        x = self.conv_block_2(x)
        x = self.dropout_2(x)
        x = self.conv_block_3(x)
        x = torch.add(x, res_x)
        out = self.norm(x)

        return out

class sSE_Block(nn.Module):
    def __init__(self, in_channels):
        super(sSE_Block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x

        return x

class TransposeBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, norm_type='GN'):
        super(TransposeBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=(0, 0), stride=2)
        self.selu = nn.SELU()
        
        if norm_type == 'BN':
          self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'GN':
          self.norm = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.selu(x)
        out = self.norm(x)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64,
                 last_dropout=0.3, norm_type='GN', do_p=0.1):
        super(DecoderBlock, self).__init__()

        self.transpose_block = TransposeBlock(in_channels, in_channels, norm_type=norm_type)
        self.enc_dropout_1 = nn.Dropout(0.3)
        self.conv_block_1 = ConvBlock(in_channels*2, out_channels, norm_type=norm_type)
        self.dropout = nn.Dropout2d(do_p)
        self.conv_block_2 = ConvBlock(out_channels, out_channels, norm_type=norm_type)
        self.sSE = sSE_Block(out_channels)
        
        if norm_type == 'BN':
          self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'GN':
          self.norm = nn.GroupNorm(32, out_channels)

        self.enc_dropout_2 = nn.Dropout(last_dropout)

    def forward(self, x, enc_x):

        res_x = self.transpose_block(x)
        x = torch.cat([res_x, self.enc_dropout_1(enc_x)], dim=1)
        x = self.conv_block_1(x)
        x = self.dropout(x)
        x = self.conv_block_2(x)
        x = torch.add(x, res_x)
        x = self.sSE(x)
        x = self.norm(x)
        out = self.enc_dropout_2(x)
        return out


class SNUNet(nn.Module):
    def __init__(self, input_channels=3, norm_type='GN'):
        super(SNUNet, self).__init__()

        self.enc_1 = EncoderBlock(input_channels, norm_type=norm_type)
        self.enc_2 = EncoderBlock(norm_type=norm_type)
        self.enc_3 = EncoderBlock(norm_type=norm_type)
        self.enc_4 = EncoderBlock(norm_type=norm_type)
        self.enc_5 = EncoderBlock(norm_type=norm_type)
        self.coding = CodingBlock(norm_type=norm_type)
        self.dec_1 = DecoderBlock(norm_type=norm_type)
        self.dec_2 = DecoderBlock(norm_type=norm_type)
        self.dec_3 = DecoderBlock(norm_type=norm_type)
        self.dec_4 = DecoderBlock(norm_type=norm_type)
        self.dec_5 = DecoderBlock(norm_type=norm_type, last_dropout=0.1)
        
        self.conv = nn.Conv2d(64, 1, kernel_size=(1, 1))
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()

    def _forward_impl(self, x): 

        x_1, x = self.enc_1(x) 
        x_2, x = self.enc_2(x)  
        x_3, x = self.enc_3(x) 
        x_4, x = self.enc_4(x) 
        x_5, x = self.enc_5(x) 

        x = self.coding(x) 

        x = self.dec_1(x, x_5)
        x = self.dec_2(x, x_4)
        x = self.dec_3(x, x_3)
        x = self.dec_4(x, x_2)
        x = self.dec_5(x, x_1)

        x = self.conv(x)
        out = self.sigmoid(x)

        return out

    def forward(self, x):
        self.step += 1
        return self._forward_impl(x)

    def generate(self, x):
        return self._forward_impl(x)

    def get_step(self):
        return self.step.data.item()

    def checkpoint(self, model_dir: Path, optimizer: torch.optim) -> None:
        print('Saving checkpoint')
        self.save(model_dir.joinpath("checkpoint.pt"), optimizer)

    def log(self, path: str, msg: str) -> None:
        with open(path, 'a') as f:
            print(msg, file=f)

    def load(self, path: Path, optimizer: torch.optim) -> None:
        checkpoint = torch.load(path, map_location='cpu')
        if "optimizer_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            # Backwards compatibility
            self.load_state_dict(checkpoint)

    def save(self, path, optimizer):
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, path)

    def num_params(self, print_out: bool = True) -> None:
        parameters = filter(lambda p: p.requires_grad, self.parameters())

        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)

