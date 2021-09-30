import torch
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import lr_scheduler
from ..datasets import *
from ..utils import *
from .ops import *
from scipy.io import wavfile
import multiprocessing as mp
import numpy as np
import timeit
import random
from random import shuffle
from tensorboardX import SummaryWriter
from .generator import *
from .discriminator import *
from .core import *
import json
import os
from torch import autograd
from scipy import signal
from segan.models.loss import *
import eval_metrics as em



#---------Imports for code from OC classification model-------------------

import sandbox.util_dsp as nii_dsp
import core_scripts.data_io.conf as nii_conf
import torch.nn.init as init


##################
## other utilities
##################
def trimf(x, params):
  """
  trimf: similar to Matlab definition
  https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle
  
  """
  if len(params) != 3:
    print("trimp requires params to be a list of 3 elements")
    sys.exit(1)
  a = params[0]
  b = params[1]
  c = params[2]
  if a > b or b > c:
    print("trimp(x, [a, b, c]) requires a<=b<=c")
    sys.exit(1)
  y = torch.zeros_like(x, dtype=nii_conf.d_dtype)
  if a < b:
    index = torch.logical_and(a < x, x < b)
    y[index] = (x[index] - a) / (b - a)
  if b < c:    
    index = torch.logical_and(b < x, x < c)              
    y[index] = (c - x[index]) / (c - b)
  y[x == b] = 1
  return y 
  
def delta(x):
  """ By default
  input
  -----
  x (batch, Length, dim)
  
  output
  ------
  output (batch, Length, dim)
  
  Delta is calculated along Length dimension
  """
  length = x.shape[1]
  output = torch.zeros_like(x)
  x_temp = F.pad(x.unsqueeze(1), (0, 0, 1, 1), # torch_nn_func changed to F
                 'replicate').squeeze(1)
  output = -1 * x_temp[:, 0:length] + x_temp[:,2:]
  return output


def linear_fb(fn, sr, filter_num):
  """linear_fb(fn, sr, filter_num)
  create linear filter bank based on trim
  input
  -----
    fn: int, FFT points
    sr: int, sampling rate (Hz)
    filter_num: int, number of filters in filter-bank
  
  output
  ------
    fb: tensor, (fn//2+1, filter_num)
  Note that this filter bank is supposed to be used on 
  spectrum of dimension fn//2+1.
  See example in LFCC.
  """
  # build the triangle filter bank
  f = (sr / 2) * torch.linspace(0, 1, fn//2+1)
  filter_bands = torch.linspace(min(f), max(f), filter_num+2)
    
  filter_bank = torch.zeros([fn//2+1, filter_num])
  for idx in range(filter_num):
    filter_bank[:, idx] = trimf(
      f, [filter_bands[idx], 
        filter_bands[idx+1], 
        filter_bands[idx+2]])
  return filter_bank

def padding(spec, ref_len,device):
  # print('hi: {}'.format(spec.shape))

  _,width, cur_len = spec.shape
  assert ref_len > cur_len
  padd_len = ref_len - cur_len
  ex_tensor = torch.zeros(spec.shape[0],width, padd_len, dtype=spec.dtype).to(device)
  device_id = ex_tensor.device
  # print("---------------------device_id: {}-------------------".format(device_id))
  return torch.cat((spec, ex_tensor), 2)

def repeat_padding(spec, ref_len):
  mul = int(np.ceil(ref_len / spec.shape[1]))
  # print("------------------spec.shape: {} mul: {}".format(spec.shape, mul))
  spec = spec.repeat(1,1, mul)[:,:, :ref_len]
  return spec        

#################
## LFCC front-end
#################

class LFCC(nn.Module):
  """ Based on asvspoof.org baseline Matlab code.
  Difference: with_energy is added to set the first dimension as energy
  """
  def __init__(self, batch_size, fl, fs, fn, sr, filter_num, feat_len, pading = 'repeat', 
         with_energy=False, with_emphasis=True,
         with_delta=True, flag_for_LFB=False, device = 'cuda'):
    """ Initialize LFCC
    
    Para:
    -----
      fl: int, frame length, (number of waveform points)
      fs: int, frame shift, (number of waveform points)
      fn: int, FFT points
      sr: int, sampling rate (Hz)
      filter_num: int, number of filters in filter-bank
      with_energy: bool, (default False), whether replace 1st dim to energy
      with_emphasis: bool, (default True), whether pre-emphaze input wav
      with_delta: bool, (default True), whether use delta and delta-delta
    
      for_LFB: bool (default False), reserved for LFB feature
    """
    super(LFCC, self).__init__()
    self.fl = fl
    self.fs = fs
    self.fn = fn
    self.sr = sr
    self.filter_num = filter_num
    self.feat_len = feat_len
    self.pading = pading
    self.batch_size = batch_size
    self.device = device
    # print("batch_size: {}".format(self.batch_size))
    
    # build the triangle filter bank
    f = (sr / 2) * torch.linspace(0, 1, fn//2+1)
    filter_bands = torch.linspace(min(f), max(f), filter_num+2)
    
    filter_bank = torch.zeros([fn//2+1, filter_num])
    for idx in range(filter_num):
      filter_bank[:, idx] = trimf(
        f, [filter_bands[idx], 
          filter_bands[idx+1], 
          filter_bands[idx+2]])
    self.lfcc_fb = nn.Parameter(filter_bank, requires_grad=False)

    # DCT as a linear transformation layer
    self.l_dct = nii_dsp.LinearDCT(filter_num, 'dct', norm='ortho')

    # opts
    self.with_energy = with_energy
    self.with_emphasis = with_emphasis
    self.with_delta = with_delta
    self.flag_for_LFB = flag_for_LFB
    return
  
  def forward(self, x):
    """
    
    input:
    ------
     x: tensor(batch, length), where length is waveform length
    
    output:
    -------
     lfcc_output: tensor(batch, dim_num, frame_num) eg: (100, 60, 103)
    """
    # pre-emphsis 
    if self.with_emphasis:
      x[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
    # print("-------------x.shape(): {}".format(x.shape))
    # STFT
    x_stft = torch.stft(x, self.fn, self.fs, self.fl, 
              window=torch.hamming_window(self.fl).to(x.device), 
              onesided=True, pad_mode="constant")        
    # amplitude
    sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
    
    # filter bank
    fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) + 
                 torch.finfo(torch.float32).eps)
    
    # DCT (if necessary, remove DCT)
    lfcc = self.l_dct(fb_feature) if not self.flag_for_LFB else fb_feature
    
    # Add energy 
    if self.with_energy:
      power_spec = sp_amp / self.fn
      energy = torch.log10(power_spec.sum(axis=2)+ 
                 torch.finfo(torch.float32).eps)
      lfcc[:, :, 0] = energy

    # Add delta coefficients
    if self.with_delta:
      lfcc_delta = delta(lfcc)
      lfcc_delta_delta = delta(lfcc_delta)
      lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
      lfcc_output = lfcc_output.permute(0, 2, 1).contiguous()
      
      this_feat_len = lfcc_output.shape[2]           
      # print("-----------lfcc_output.shape :{}".format(lfcc_output.shape))

      if this_feat_len > self.feat_len:
        startp = np.random.randint(this_feat_len-self.feat_len)
        lfcc_output = lfcc_output[:,:, startp:startp+self.feat_len]
      if this_feat_len < self.feat_len:
        if self.pading == 'zero':
          # print(type(feat_len))
          lfcc_output = padding(lfcc_output, self.feat_len,self.device)
          # print("-------------",lfcc_output.shape)
          # print("new: {}".format(feat_mat.unsqueeze(0).shape))
        elif self.pading == 'repeat':
          lfcc_output = repeat_padding(lfcc_output, self.feat_len)
          # print("new: {}".format(feat_mat.shape))
        else:
          raise ValueError('Padding should be zero or repeat!')

      lfcc_output = lfcc_output.unsqueeze(1).float()
      # print("-----------lfcc_output.shape :{}".format(lfcc_output.shape))


    else:
      lfcc_output = lfcc
      lfcc_output = lfcc_output.permute(0, 2, 1).contiguous()
      this_feat_len = lfcc_output.shape[2]           
      print("-----------lfcc_output.shape :{}".format(lfcc_output.shape))
      if this_feat_len > self.feat_len:
        startp = np.random.randint(this_feat_len-self.feat_len)
        lfcc_output = lfcc_output[:,:, startp:startp+self.feat_len]
      if this_feat_len < self.feat_len:
        if self.pading == 'zero':
          # print(type(feat_len))
          lfcc_output = padding(lfcc_output, self.feat_len,self.device)
          # print("-------------",lfcc_output)
          # print("new: {}".format(feat_mat.unsqueeze(0).shape))
        elif self.pading == 'repeat':
          lfcc_output = repeat_padding(lfcc_output, self.feat_len)
          # print("new: {}".format(feat_mat.shape))
        else:
          raise ValueError('Padding should be zero or repeat!')

      lfcc_output = lfcc_output.unsqueeze(1).float()
      # print("-----------lfcc_output.shape :{}".format(lfcc_output.shape))
    return lfcc_output


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)

    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])

    return y

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, pading = 'repeat', in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50, feat_len = 750):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.feat_len = feat_len

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                    self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        out = F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)
        # print("-----------out.shape :{}".format(out.shape))
        this_feat_len = out.shape[2]           

        if this_feat_len > self.feat_len:
          startp = np.random.randint(this_feat_len-self.feat_len)
          out = out[:,:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
          if self.pading == 'zero':
            # print(type(feat_len))
            out = padding(out, self.feat_len,self.device)
            # print("-------------",lfcc_output.shape)
            # print("new: {}".format(feat_mat.unsqueeze(0).shape))
          elif self.pading == 'repeat':
            out = repeat_padding(out, self.feat_len)
            # print("new: {}".format(feat_mat.shape))
          else:
            raise ValueError('Padding should be zero or repeat!')
        # print("-----------out.shape :{}".format(out.shape))


        return out.unsqueeze(1)


#---------------- /media/rohit/NewVolume/codes/SA/PycharmProjects/ASVSpoof2021/AIR-ASVSpoof_LFCC_layer/resnet.py------------------------ 

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        #self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):

        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0)==1:
            attentions = F.softmax(torch.tanh(weights),dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5*torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = torch.cat((avg_repr,std_repr),1)

            return representations


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }

def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False

class LFCC_model(nn.Module):
    def __init__(self, num_nodes, enc_dim, batch_size, frame_length, \
        frame_shift,fft_points, sr, filter_num, feat_len, resnet_type='18', nclasses=2, pad_type = 'zero', sinc = False, device = 'cuda'):
        self.in_planes = 16
        super(LFCC_model, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fft_points = fft_points
        self.filter_num = filter_num
        self.feat_len = feat_len
        self.pad_type = pad_type
        self.sinc = sinc

        self.front_end_LFCC = LFCC(batch_size,frame_length,frame_shift,fft_points,sr,filter_num,feat_len, pading = self.pad_type,device = 'cuda')
        self.front_end_Sincnet = SincConv_fast(60, 251, 16000, pading = self.pad_type)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = SelfAttention(256)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("------------------------------",)
        if self.sinc == False:
          x = self.front_end_LFCC(x)
          # print("----------------------------x.shape: {}".format(x.shape))
        else:
          x = x.unsqueeze(1)
          x = self.front_end_Sincnet(x)
        # print("----------------------------x.shape: {}".format(x.shape))
        x = self.conv1(x)
        x = self.act(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        # print("----------x.size: {}".format(x.size()))
        x = self.act(self.bn5(x)).squeeze(2)
        # print("----------x.size: {}".format(x.size()))
        x = x.permute(0, 2, 1).contiguous()
        # print("----------x.size: {}".format(x.size()))
        stats = self.attention(x)

        feat = self.fc(stats)

        mu = self.fc_mu(feat)

        return feat, mu
#------------------------------Code from OC OVER-------------------------------------------