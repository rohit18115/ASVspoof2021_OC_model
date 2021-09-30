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
from oc.models.loss import *
import eval_metrics as em
from tqdm import tqdm
import gc 
import torchaudio
# import torchaudio.functional as F
import torchaudio.transforms as T
import torch_dct as dct






#---------Imports for code from OC classification model-------------------

import sandbox.util_dsp as nii_dsp
import core_scripts.data_io.conf as nii_conf
import torch.nn.init as init




# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        print('Initializing weights of convresblock to 0.0, 0.02')
        for k, p in m.named_parameters():
            if 'weight' in k and 'conv' in k:
                p.data.normal_(0.0, 0.02)
    elif classname.find('Conv1d') != -1:
        print('Initialzing weight to 0.0, 0.02 for module: ', m)
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            print('bias to 0 for module: ', m)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        print('Initializing FC weight to xavier uniform')
        nn.init.xavier_uniform_(m.weight.data)

def wsegan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        print('Initializing weights of convresblock to 0.0, 0.02')
        for k, p in m.named_parameters():
            if 'weight' in k and 'conv' in k:
                nn.init.xavier_uniform_(p.data)
    elif classname.find('Conv1d') != -1:
        print('Initialzing weight to XU for module: ', m)
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('ConvTranspose1d') != -1:
        print('Initialzing weight to XU for module: ', m)
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        print('Initializing FC weight to XU')
        nn.init.xavier_uniform_(m.weight.data)

def z_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        # let it active
        m.train()
    else:
        m.eval()



class OC_model(Model):

    def __init__(self, opts, name='OC_model',
                 lfcc_model = None, 
                 AMS_loss = None,
                 OCS_loss = None,
                 MVS_loss = None):
        super(OC_model, self).__init__(name)
        self.save_path = opts.save_path
        self.preemph = opts.preemph
        if lfcc_model is None:
            self.OC = LFCC_model(3, opts.enc_dim_oc, opts.batch_size, opts.frame_length_oc, \
            opts.frame_shift_oc,opts.fft_points_oc, opts.sr_oc, opts.filter_num_oc, opts.feat_len_oc, \
            resnet_type='18', nclasses=2, pad_type = opts.pad_type_oc,cmvn = opts.cmvn, frontend_name = opts.frontend_name)
        else:
            self.OC = lfcc_model
        # self.OC.apply(weights_init) # will comment this for now, will uncomment and make changes if feel that performance is not up to the mark
        print("LFCC_model:", self.OC)

        if opts.add_loss_oc == "amsoftmax":         
            if AMS_loss is None:
                self.AMS = AMSoftmax(2, opts.enc_dim_oc, s=opts.alpha_oc, m=opts.r_real_oc)
            else:
                self.AMS = AMS_loss

        if opts.add_loss_oc == "ocsoftmax":
            if OCS_loss is None:
                self.OCS = OCSoftmax(opts.enc_dim_oc, r_real=opts.r_real_oc, r_fake=opts.r_fake_oc, alpha=opts.alpha_oc, loss_type=opts.loss_type)
            else:
                self.OCS = OCS_loss
        if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
            if MVS_loss is None:
                self.MVS = MVSoftmax(fc_type=opts.add_loss_oc, margin=opts.r_real_oc, t=0.2, scale=opts.alpha_oc, embedding_size=opts.enc_dim_oc, num_class=2, easy_margin=False)
            else:
                self.MVS = MVS_loss
    def adjust_learning_rate(self, opts, optimizer, epoch_num):
        learn_r_oc = opts.lr_oc * (opts.lr_decay_oc ** (epoch_num // opts.interval_oc))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learn_r_oc
    def build_optimizers(self, opts):

        
        OCopt = optim.Adam(self.OC.parameters(), lr=opts.lr_oc,
                                  betas=(opts.beta_1_oc, opts.beta_2_oc), eps=opts.eps_oc, weight_decay=0.0005)
        AMSopt = ''
        OCSopt = ''
        MVSopt = ''
        if opts.add_loss_oc == "amsoftmax":
            AMSopt = optim.SGD(self.AMS.parameters(), lr=0.01)

        if opts.add_loss_oc == "ocsoftmax":
            OCSopt = optim.SGD(self.OCS.parameters(), lr=opts.lr_oc)
        if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
            MVSopt = optim.SGD(self.MVS.parameters(), lr=0.01)

        
        return MVSopt, OCopt, AMSopt, OCSopt

    def train(self, opts, dloader, criterion_oc, log_freq, val_log_freq, va_dloader=None, device='cpu'):

        self.writer = SummaryWriter(os.path.join(self.save_path, 'train'))
        monitor_loss = opts.add_loss_oc
        
        
        # Build the optimizers
        __, OCopt, _, __ = self.build_optimizers(opts)

        if opts.add_loss_oc == "amsoftmax":
            __ ,OCopt, AMSopt, _ = self.build_optimizers(opts)

        if opts.add_loss_oc == "ocsoftmax":
            __ , OCopt, _, OCSopt = self.build_optimizers(opts)
        
        if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
            MVSopt, OCopt, __, _ = self.build_optimizers(opts)


        # attach opts to models so that they are saved altogether in ckpts
        self.OC.optim = OCopt

        if opts.add_loss_oc == "amsoftmax":
          self.AMS.optim = AMSopt
        if opts.add_loss_oc == "ocsoftmax":
          self.OCS.optim = OCSopt
        if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
            self.MVS.optim = MVSopt

        
        # Build savers for end of epoch, storing up to 3 epochs each
        
        eoe_oc_saver = Saver(self.OC, opts.save_path, max_ckpts=None,
                            optimizer=self.OC.optim, prefix='EOE_OC-')
        if opts.add_loss_oc == "amsoftmax":
            eoe_ams_saver = Saver(self.AMS, opts.save_path, max_ckpts=None,
                            optimizer=self.AMS.optim, prefix='EOE_AMS-')
        if opts.add_loss_oc == "ocsoftmax":
            eoe_ocs_saver = Saver(self.OCS, opts.save_path, max_ckpts=None,
                            optimizer=self.OCS.optim, prefix='EOE_OCS-')
        if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
            eoe_mvs_saver = Saver(self.MVS, opts.save_path, max_ckpts=None,
                            optimizer=self.MVS.optim, prefix='EOE_MVS-')

        prev_val_eer_oc = 1e8
        best_val_eer_oc = 1e8
        best_train_eer_oc = 1e8
        prev_train_eer_oc = 1e8
        timings = []
        iteration = 1
# #-----------------------in current dev set the dataloader extracts random 1 second samples and pushes it to the model which might not tell the 
# #----------------------- exact performance of the model and its better to implement the validation part as it is done in the test_OC code as it will 
# #-----------------------help the model judge better
#         filehandle = open(opts.protocol_address_val, 'r')
#         protocol = []
#         while True:
#             # read a single line

#             line = (filehandle.readline())

#             protocol.append(line)

#             if not line:
#                 break

#         # lab_batch = np.zeros(batch_size)

#             # close the pointer to that file
#         filehandle.close()

#         protocol = [s[:-1] for s in protocol]

#         tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
#                       "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
#                       "A19": 19}
        

#         protocol = pd.DataFrame([s.strip().split(' ') for s in protocol])

#         protocol.columns = ['speaker_id', 'clean_abs', 'blah', 'system_id', 'label', 'noisy_abs', 'degradation_type']

#         protocol = protocol[['speaker_id', 'clean_abs', 'system_id', 'label', 'noisy_abs', 'degradation_type']]

#         protocol.dropna(inplace=True)

#         protocol.drop_duplicates(subset="clean_abs", keep='first', inplace=True)

        for epoch in tqdm(range(1, opts.epoch + 1)):
            beg_t = timeit.default_timer()
            trainlossDict = defaultdict(list)
            devlossDict = defaultdict(list)
            self.OC.train()
            if opts.add_loss_oc == "amsoftmax":
                self.AMS.train()
            if opts.add_loss_oc == "ocsoftmax":
                self.OCS.train()
            if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                self.MVS.train()

            self.adjust_learning_rate(opts, self.OC.optim, epoch)
            if opts.add_loss_oc == "ocsoftmax":
                self.adjust_learning_rate(opts, self.OCS.optim, epoch)
            elif opts.add_loss_oc == "amsoftmax":
                self.adjust_learning_rate(opts, self.AMS.optim, epoch)
            elif opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                self.adjust_learning_rate(opts, self.MVS.optim, epoch)
            idx_loader_oc, score_loader_oc = [], []
            for bidx, batch in enumerate(tqdm(dloader), start=1):
                gc.collect()
                if len(batch) == 7:
                    uttname, clean, noisy, _, _, labels_oc, degradation_type = batch
                    # print("uttname: {}, label: {}, degradation_type: {}".format(uttname,labels_oc,degradation_type)) # checks for understanding code
                else:
                    raise ValueError('Returned {} elements per '
                                     'sample?'.format(len(sample)))

                # clean = clean.unsqueeze(1)
                # noisy = noisy.unsqueeze(1)
                # print("------------------clean.size:{} and c_lengths.shape:{}".format(clean.size(),len(c_lengths))) # checks for understanding code
                clean = clean.to(device)
                # print("clean.shape:{} labels_oc length:{} labels_oc:{}".format(clean.shape, len(labels_oc), labels_oc))
                noisy = noisy.to(device)
                # system_id = system_id.to(device)
                labels_oc = labels_oc.to(device)
                # print("----------------------labels_oc.size(): {}".format(labels_oc.size()))

                feats_oc, lfcc_outputs = self.OC(noisy)
                lfcc_loss = criterion_oc(lfcc_outputs, labels_oc)
                score_oc = F.softmax(lfcc_outputs, dim=1)[:, 0]
                if opts.add_loss_oc == "ocsoftmax":
                    ocsoftmaxloss, score_oc = self.OCS(feats_oc, labels_oc)
                    lfcc_loss = ocsoftmaxloss * opts.weight_loss_oc
                if opts.add_loss_oc == "amsoftmax":
                    outputs_oc, moutputs_oc = self.AMS(feats_oc, labels_oc)
                    if opts.loss_type == None:
                        lfcc_loss = criterion_oc(moutputs_oc, labels_oc)
                    elif opts.loss_type == 'HardMining':
                        lfcc_loss = loss_final(moutputs_oc, labels_oc, opts.loss_type)
                    score_oc = F.softmax(outputs_oc, dim=1)[:, 0]
                elif opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                    moutputs_oc, outputs_oc = self.MVS(feats_oc, labels_oc)
                    score_oc = outputs_oc.squeeze(1)
                    if opts.loss_type == None:
                        lfcc_loss = criterion_oc(moutputs_oc, labels_oc)
                    elif opts.loss_type == 'HardMining':
                        lfcc_loss = loss_final(moutputs_oc, labels_oc, opts.loss_type)
                    
                idx_loader_oc.append(labels_oc.detach().cpu())
                score_loader_oc.append(score_oc.detach().cpu())

                if opts.add_loss_oc == "softmax":
                    OCopt.zero_grad()
                    trainlossDict[opts.add_loss_oc].append(lfcc_loss.item())
                if opts.add_loss_oc == "ocsoftmax":
                    OCopt.zero_grad()
                    OCSopt.zero_grad()
                    trainlossDict[opts.add_loss_oc].append(ocsoftmaxloss.item())
                if opts.add_loss_oc == "amsoftmax":
                    OCopt.zero_grad()
                    AMSopt.zero_grad()
                    trainlossDict[opts.add_loss_oc].append(lfcc_loss.item())
                if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                    OCopt.zero_grad()
                    MVSopt.zero_grad()
                    trainlossDict[opts.add_loss_oc].append(lfcc_loss.item())

                lfcc_loss.backward()

                OCopt.step()
                if opts.add_loss_oc == "ocsoftmax":
                    OCSopt.step()
                if opts.add_loss_oc == "amsoftmax":
                    AMSopt.step()
                if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                    MVSopt.step()

                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    
                    if opts.add_loss_oc == "softmax":
                        lfcc_loss_v = lfcc_loss.cpu().detach().item()
                    if opts.add_loss_oc == "ocsoftmax":
                        ocsoftmaxloss_v = ocsoftmaxloss.cpu().detach().item()
                    if opts.add_loss_oc == "amsoftmax" or opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                        lfcc_loss_v = lfcc_loss.cpu().detach().item()

                    log = '(Iter {}) Batch {}/{} (Epoch {}) '.format(iteration, bidx,
                                                   len(dloader), epoch)
                    if opts.add_loss_oc == "softmax":
                        log += 'lfcc_l:{:.4f} btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(lfcc_loss_v, 
                                     timings[-1],
                                     np.mean(timings))
                    if opts.add_loss_oc == "ocsoftmax":
                        log += 'ocs_l:{:.4f} btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(ocsoftmaxloss_v,
                                     timings[-1],
                                     np.mean(timings))
                    if opts.add_loss_oc == "amsoftmax" or opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                        log += 'ams_l:{:.4f} btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(lfcc_loss_v,
                                     timings[-1],
                                     np.mean(timings))
                    
                    # print(log)
                    if opts.add_loss_oc == "softmax":
                        self.writer.add_scalar('lfcc_l', lfcc_loss_v,
                                            iteration)
                    if opts.add_loss_oc == "ocsoftmax":
                        self.writer.add_scalar('ocs_l', ocsoftmaxloss_v,
                                            iteration)
                    if opts.add_loss_oc == "amsoftmax":
                        self.writer.add_scalar('ams_l', lfcc_loss_v,
                                            iteration)
                    if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                        self.writer.add_scalar('mvs_l', lfcc_loss_v,
                                            iteration)
                    self.writer.add_histogram('lfcc_feats', feats_oc.cpu().data,
                                              iteration, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data,
                                              iteration, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.detach().cpu().data,
                                              iteration, bins='sturges')
            
            scores_oc = torch.cat(score_loader_oc, 0).data.cpu().numpy()
            labels_oc = torch.cat(idx_loader_oc, 0).data.cpu().numpy()
            array_sum = np.sum(scores_oc)
            array_has_nan = np.isnan(array_sum)
            # print("--------------labels_oc : {}".format(labels_oc))
            # print("--------------uttname : {}".format(uttname))
            train_eer_oc = em.compute_eer(scores_oc[labels_oc == 0], scores_oc[labels_oc == 1])[0]
            other_train_eer_oc = em.compute_eer(-scores_oc[labels_oc == 0], -scores_oc[labels_oc == 1])[0]
            # print('train_eer_oc: {}, other_train_eer_oc: {}, , scores contains None: {}'.format(train_eer_oc, other_train_eer_oc, array_has_nan))
            train_eer_oc = min(train_eer_oc, other_train_eer_oc)
            with open(os.path.join(opts.save_path, "train_loss.log"), "a") as log:
                log.write(str(epoch)  + "\t" +
                          str(np.nanmean(trainlossDict[monitor_loss])) +  "\t" + str(train_eer_oc*100) + "\n")
            print('train_eer_oc %: {}, , scores contains None: {}'.format(train_eer_oc*100, array_has_nan))
            if train_eer_oc < best_train_eer_oc:
                print('best train_EER %: {}  Prev train_EER %: {} Prev best train_EER %: {}'.format(train_eer_oc*100, prev_train_eer_oc*100, best_train_eer_oc*100))
                best_train_eer_oc = train_eer_oc
                prev_train_eer_oc = train_eer_oc
            else:
                print('Current train_EER %: {}  Prev train_EER %: {} Best train_EER %: {}'.format(train_eer_oc*100, prev_train_eer_oc*100, best_train_eer_oc*100))
                train_prev_eer_oc = train_eer_oc

            if va_dloader is not None:
                
                val_eer_oc_ = self.evaluate(opts, va_dloader,  
                                                     val_log_freq, iteration, criterion_oc, devlossDict, epoch, monitor_loss, do_noisy=True)
                
                
                self.writer.add_scalar('Val_EER %:', val_eer_oc_,epoch*100)
                if val_eer_oc_ < best_val_eer_oc:
                    
                    print('best_Val_EER %: {}  Prev_Val_EER %: {}, prev_best_Val_EER %:{}'.format(val_eer_oc_*100, prev_val_eer_oc*100, best_val_eer_oc*100))
                    best_val_eer_oc = val_eer_oc_
                    prev_val_eer_oc = val_eer_oc_
                    # best_val_obj = val_obj
                    patience = opts.patience
                    # save models with true valid curve is minimum
                    
                    self.OC.save(self.save_path,iteration, True)
                    if opts.add_loss_oc == "ocsoftmax":
                        self.OCS.save(self.save_path, iteration, True)
                    elif opts.add_loss_oc == "amsoftmax":
                        self.AMS.save(self.save_path, iteration, True)
                    elif opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                        self.MVS.save(self.save_path, iteration, True)
                else:
                    patience -= 1
                    print('Val loss did not improve. Patience'
                          '{}/{}'.format(patience,
                                         opts.patience))
                    if patience <= 0:
                        print('STOPPING SEGAN TRAIN: OUT OF PATIENCE.')
                        break
                    
                    print('Cureent_Val_EER %: {}  Prev_Val_EER %: {}, best_Val_EER %:{}'.format(val_eer_oc_*100, prev_val_eer_oc*100, best_val_eer_oc*100))
                    prev_val_eer_oc = val_eer_oc_
                    # best_val_obj = val_obj
                    
            # save models in end of epoch with EOE savers
            
            self.OC.save(self.save_path, iteration, saver = eoe_oc_saver)
            if opts.add_loss_oc == "ocsoftmax":
                self.OCS.save(self.save_path, iteration, saver = eoe_ocs_saver)
            if opts.add_loss_oc == "amsoftmax":
                self.AMS.save(self.save_path, iteration, saver = eoe_ams_saver)
            if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                self.MVS.save(self.save_path, iteration, saver = eoe_mvs_saver)
            iteration += 1 
    
    def evaluate(self, opts, dloader, log_freq, iteration, criterion_oc, devlossDict, epoch, monitor_loss, do_noisy=False,
                 max_samples=8, device='cuda'):
        """ Objective evaluation with PESQ, SSNR, COVL, CBAK and CSIG """
        
        self.OC.eval()
        if opts.add_loss_oc == "amsoftmax":
            self.AMS.eval()
        if opts.add_loss_oc == "ocsoftmax":
            self.OCS.eval()
        if opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
            self.MVS.eval()

        with torch.no_grad():
            idx_loader_oc, score_loader_oc = [], []            
            # going over dataset ONCE
            for bidx, batch in enumerate(tqdm(dloader), start=1):
                sample = batch
                if len(sample) == 7:
                    # uttname, clean, noisy, slice_idx = batch
                    _, val_clean, val_noisy, _, _, labels_oc, _ = batch
                else:
                    raise ValueError('Returned {} elements per '
                                     'sample?'.format(len(sample)))
                # val_clean = val_clean.to(device)
                val_noisy = val_noisy.to(device)

                
                labels_oc = labels_oc.to(device)
                # degradation_type = degradation_type.to(device)
                


                feats_oc, lfcc_outputs = self.OC(val_noisy)

                lfcc_loss = criterion_oc(lfcc_outputs, labels_oc)
                
                score_oc = F.softmax(lfcc_outputs, dim=1)[:, 0]

                if opts.add_loss_oc == "softmax":
                    devlossDict[opts.add_loss_oc].append(lfcc_loss.item())
                if opts.add_loss_oc == "amsoftmax":
                    outputs_oc, moutputs_oc = self.AMS(feats_oc, labels_oc)
                    lfcc_loss = criterion_oc(moutputs_oc, labels_oc)
                    score_oc = F.softmax(outputs_oc, dim=1)[:, 0]
                    devlossDict[opts.add_loss_oc].append(lfcc_loss.item())
                elif opts.add_loss_oc == "ocsoftmax":
                    ocsoftmaxloss, score_oc = self.OCS(feats_oc, labels_oc)
                    devlossDict[opts.add_loss_oc].append(ocsoftmaxloss.item())
                elif opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc":
                    moutputs_oc, score_oc = self.MVS(feats_oc, labels_oc)
                    score_oc = score_oc.squeeze(1)
                    print(score_oc)
                    lfcc_loss = criterion_oc(moutputs_oc, labels_oc)
                    devlossDict[opts.add_loss_oc].append(lfcc_loss.item())

                idx_loader_oc.append(labels_oc.detach())
                score_loader_oc.append(score_oc.detach())


                # print("bidx: {}".format(bidx))
                # print("--------------uttname : {}".format(uttname))
                # print("--------------labels_oc : {}".format(labels_oc))
                # print("--------------genh.size : {}, clean.size: {}".format(Genh.shape, \
                # clean.shape))
#---------------------------------------------------------------------------
        scores_oc = torch.cat(score_loader_oc, 0).data.cpu().numpy()
        labels_oc = torch.cat(idx_loader_oc, 0).data.cpu().numpy()
        array_sum = np.sum(scores_oc)
        array_has_nan = np.isnan(array_sum)
        # print("--------------labels_oc : {}".format(labels_oc))
        # print("--------------uttname : {}".format(uttname))

#--------------------------------------validation method according to the test_oc.py-------------
            # idx_loader_oc, score_loader_oc = np.array([]), np.array([]) 
            # spoof_genuine = {'spoof': 1, 'bonafide':0}
            # for i in tqdm(range(len(protocol))):
                
            #     file_id = protocol.iloc[i, 4]
            #     # print(file_id)
            #     degradation_type = protocol.iloc[i,5]
            #     name = protocol.iloc[i,1].split('/')[-1]
                
            #     [signal, fs] = sf.read(file_id)
            #     signal = pre_emphasize(signal, opts.preemph)
            #     # signal = normalize_wave_minmax(signal) #this line makes audio inaudioble but i think it is present in the dataloader so it might improve result of the model

            #     beg_samp = 0
            #     end_samp = opts.slice_size
            #     wshift = int(opts.data_stride * opts.slice_size)
            #     wlen = int(opts.slice_size)
            #     # Batch_dev = args.batch_size
            #     N_fr = max(math.ceil((signal.shape[0] - wlen) / (wshift)), 0)
            #     # print("--------------N_fr:{}".format(N_fr+1))
                
            #     signal = torch.from_numpy(signal)
            #     sig_arr = torch.zeros([N_fr + 1, wlen])
            #     pout = torch.zeros(N_fr + 1)
                
            #     signal = signal.cuda().float().contiguous()
            #     sig_arr = sig_arr.cuda().float().contiguous()
            #     pout = pout.cuda().float().contiguous()
            #     count_fr = 0
            #     # print("Name: {} siganl.shape: {}".format(name,signal.shape))
                
            #     while end_samp < signal.shape[0]:
            #         # print("------------beg_sample : {}, end_sample: {}".format(beg_samp,end_samp))
            #         sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            #         beg_samp = beg_samp + wshift
            #         end_samp = beg_samp + wlen
            #         # print("------------beg_sample : {}, end_sample: {}".format(beg_samp,end_samp))
            #         count_fr = count_fr + 1
            #     if end_samp > signal.shape[0]:
            #         # print("count_fr: {}".format(count_fr))
            #         # print("sig_arr.shape:{}".format(sig_arr.shape))
            #         # print("-------------Name: {} signal.shape: {}".format(name,signal.shape)) 
            #         # print("-------------signal[beg_samp:signal.shape[0]].shape[0]:{}".format(signal[beg_samp:signal.shape[0]].shape[0]))
            #         # print("----------------------(signal.shape[0]- beg_samp):{}".format((signal.shape[0] - beg_samp)))
            #         # assert signal[beg_samp:signal.shape[0]].shape[0] == (signal.shape[0] - beg_samp), print("damn")
            #         sig_arr[count_fr, :(signal.shape[0] - beg_samp)] = signal[beg_samp:signal.shape[0]]
            #     # print("sig_arr.shape:{}".format(sig_arr.shape))
                
                    

            #     if count_fr >= 0:
            #         # j = j + 1
            #         # # print(j)

            #         inp = sig_arr
            #         # print("------inp.size out", inp.shape)

            #         feats, lfcc_outputs = self.OC(inp)
                    
            #         lab_batch = spoof_genuine[protocol.iloc[i, 3]]
            #         labels = [lab_batch for s in range(feats.size(0))] 
                    

            #         score = F.softmax(lfcc_outputs)[:, 0]

            #         if opts.add_loss_oc == "ocsoftmax":
            #             ang_isoloss, score = self.OCS(feats, labels)
            #         elif opts.add_loss_oc == "amsoftmax":
            #             outputs, moutputs = self.AMS(feats, labels)
            #             score = F.softmax(outputs, dim=1)[:, 0]
            
            #     pout[0:] = score
            #     score_oc_oc = torch.sum((pout[:]), dim=0) / len(pout)
            #     idx_loader_oc = np.append(idx_loader_oc ,lab_batch)
            #     score_loader_oc = np.append(score_loader_oc ,score_oc_oc.cpu())
            # scores_oc = score_loader_oc
            # labels_oc = idx_loader_oc
            # # scores_oc = torch.cat(score_loader_oc, 0).data.cpu().numpy()
            # # labels_oc = torch.cat(idx_loader_oc, 0).data.cpu().numpy()
            # array_sum = np.sum(scores_oc)
            # array_has_nan = np.isnan(array_sum)
#--------------------------------------------------------------------------------------

        val_eer_oc = em.compute_eer(scores_oc[labels_oc == 0], scores_oc[labels_oc == 1])[0]
        other_val_eer_oc = em.compute_eer(-scores_oc[labels_oc == 0], -scores_oc[labels_oc == 1])[0]
        # print('val_eer_oc: {}, other_val_eer_oc: {}, , scores contains None: {}'.format(val_eer_oc, other_val_eer_oc, array_has_nan))
        val_eer_oc = min(val_eer_oc, other_val_eer_oc)
        with open(os.path.join(opts.save_path, "dev_loss.log"), "a") as log:
                    log.write(str(epoch) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer_oc*100) +"\n")
        print('val_eer_oc %: {} , scores contains None: {}'.format(val_eer_oc*100, array_has_nan))
        

            
        return val_eer_oc




#----------------------------------------------code copied from OC Classification model:------------------------------------------------

#--------------------/media/rohit/NewVolume/codes/SA/PycharmProjects/ASVSpoof2021/AIR-ASVSpoof_LFCC_layer/LFCC_layer.py----------------

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
## Spectrogram (FFT) front-end
#################


class MelSpec2DDCT(nn.Module):
    """ Spectrogram front-end
    """
    def __init__(self, fl, fs, fn, sr, filter_num, feat_len, padding='repeat', 
                 with_energy=True,
                 with_emphasis=True, device = 'cuda'):
        """ Initialize Spectrogram
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default False), whether use delta and delta-delta
        
        """
        super(MelSpec2DDCT, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.feat_len = feat_len
        self.padding = padding
        self.filter_num = filter_num
        # opts
        self.energy = {0:False, 1:True}
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        
        self.spectrogram = T.MelSpectrogram(n_fft=self.fn,
                                    sample_rate = self.sr,
                                    win_length=self.fl,
                                    hop_length=self.fs,
                                    n_mels = 128,
                                    normalized = True,
                                    # mel_scale =  'htk',
                                    # center=True,
                                    # pad_mode="reflect",
                                    power= self.energy[self.with_energy],
        )
        
        self.melspec_dct = nii_dsp.LinearDCT(128 , 'dct', norm='ortho')

        return
    
    def forward(self, x):
        """
        
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis 
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        spec = self.spectrogram(x)
        print("spec.size():{}".format(spec.size()))
        spec = spec.permute(0,2,1).contiguous()
        sp_output = self.melspec_dct(spec)
        print("sp_output.size(): {}".format(sp_output.size()))
        
        # # STFT
        # x_stft = torch.stft(x, self.fn, self.fs, self.fl, 
        #                     window=torch.hamming_window(self.fl).to(x.device), 
        #                     onesided=True, pad_mode="constant")        
        # # amplitude
        # sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        # print("Before delta(Spectrogrma): {}".format(sp_amp.size()))
        # # Add delta coefficients
        # if self.with_delta:
        #     sp_delta = delta(sp_amp)
        #     sp_delta_delta = delta(sp_delta)
        #     sp_output = torch.cat((sp_amp, sp_delta, sp_delta_delta), 2)
        #     sp_output = sp_output.permute(0,2,1).contiguous()
        #     this_feat_len = sp_output.shape[2]

        #     if this_feat_len > self.feat_len:
        #         startp = np.random.randint(this_feat_len-self.feat_len)
        #         sp_output = sp_output[:,:, startp:startp+self.feat_len]
        #     if this_feat_len < self.feat_len:
        #         if self.padding == 'zero':
        #         # print(type(feat_len))
        #             sp_output = padding(sp_output, self.feat_len,self.device)
        #         # print("-------------",lfcc_output.shape)
        #         # print("new: {}".format(feat_mat.unsqueeze(0).shape))
        #         elif self.padding == 'repeat':
        #             sp_output = repeat_padding(sp_output, self.feat_len)
        #         # print("new: {}".format(feat_mat.shape))
        #         else:
        #             raise ValueError('Padding should be zero or repeat!')

        #     sp_output = sp_output.unsqueeze(1).float()

        # else:
        #     sp_output = sp_amp
        #     sp_output = sp_output.permute(0,2,1).contiguous()
        #     this_feat_len = sp_output.shape[2]

        #     if(this_feat_len>self.feat_len):
        #         startp = np.random.randint(this_feat_len-self.feat_len)
        #         sp_output = sp_output[:,:, startp:startp+self.feat_len]                
        #     if this_feat_len < self.feat_len:
        #         if self.padding == 'zero':
        #             # print(type(feat_len))
        #             sp_output = padding(sp_output, self.feat_len,self.device)
        #             # print("-------------",lfcc_output)
        #             # print("new: {}".format(feat_mat.unsqueeze(0).shape)
        #         if self.padding == 'repeat':
        #             sp_output = repeat_padding(sp_output, self.feat_len)
        #             # print("new: {}".format(feat_mat.shape))
        #         else:
        #             raise ValueError('Padding should be zero or repeat!')
            
        sp_output = sp_output.unsqueeze(1).float()
                
        # done
        return sp_output

#################
## MFCC front-end
#################

class MFCC(nn.Module):
  """ Based on asvspoof.org baseline Matlab code.
  Difference: with_energy is added to set the first dimension as energy
  """
  def __init__(self, batch_size, fl, fs, fn, sr, filter_num, feat_len, pading = 'repeat', 
         with_energy=True, with_emphasis=True,
         with_delta=True, device = 'cuda'):
    """ Initialize MFCC
    
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
    super(MFCC, self).__init__()
    self.fl = fl
    self.fs = fs
    self.fn = fn
    self.sr = sr
    self.filter_num = filter_num
    self.feat_len = feat_len
    self.pading = pading
    self.batch_size = batch_size
    self.device = device
    self.energy = {0:False, 1:True}
    self.with_energy = with_energy
    self.with_emphasis = with_emphasis
    self.with_delta = with_delta
    # self.flag_for_LFB = flag_for_LFB
    # print("batch_size: {}".format(self.batch_size))
    self.mfcc_transform = T.MFCC(sample_rate=self.sr, n_mfcc=self.filter_num,melkwargs={'n_fft': self.fn,
                                                                                'win_length': self.fl,
                                                                              'n_mels': self.fn//2+1,
                                                                              'hop_length': self.fs,
                                                                              # 'mel_scale': 'htk',
                                                                              'power': self.energy[self.with_energy],
                                                                              "normalized": True,
                                                                            }
                                                                        )
    # delta_transform = T.ComputeDeltas()
    

    # opts
    
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
    mfcc = self.mfcc_transform(x)
    mfcc = mfcc.permute(0,2,1).contiguous()
    # print("mfcc.size():{}".format(mfcc.size()))
    # print("-----------------hiiiiii")
    # Add delta coefficients
    if self.with_delta:
      mfcc_delta = delta(mfcc)
      mfcc_delta_delta = delta(mfcc_delta)
      mfcc_output = torch.cat((mfcc, mfcc_delta, mfcc_delta_delta), 2)
      mfcc_output = mfcc_output.permute(0, 2, 1).contiguous()
      this_feat_len = mfcc_output.shape[2] 
      # print("mfcc_output.size(): {}this_feat_len: {}, self.feat_len:{}".format(mfcc_output.size(), this_feat_len, self.feat_len))          

      if this_feat_len > self.feat_len:
        startp = np.random.randint(this_feat_len-self.feat_len)
        mfcc_output = mfcc_output[:,:, startp:startp+self.feat_len]
      if this_feat_len < self.feat_len:
        if self.pading == 'zero':
          # print(type(feat_len))
          mfcc_output = padding(mfcc_output, self.feat_len,self.device)
          # print("-------------",lfcc_output.shape)
          # print("new: {}".format(feat_mat.unsqueeze(0).shape))
        elif self.pading == 'repeat':
          mfcc_output = repeat_padding(mfcc_output, self.feat_len)
          # print("new: {}".format(feat_mat.shape))
        else:
          raise ValueError('Padding should be zero or repeat!')

      mfcc_output = mfcc_output.unsqueeze(1).float()

    else:
      mfcc_output = mfcc
      mfcc_output = mfcc_output.permute(0, 2, 1).contiguous()
      this_feat_len = mfcc_output.shape[2]           

      if this_feat_len > self.feat_len:
        startp = np.random.randint(this_feat_len-self.feat_len)
        mfcc_output = mfcc_output[:,:, startp:startp+self.feat_len]
      if this_feat_len < self.feat_len:
        if self.pading == 'zero':
          # print(type(feat_len))
          mfcc_output = padding(mfcc_output, self.feat_len,self.device)
          # print("-------------",lfcc_output)
          # print("new: {}".format(feat_mat.unsqueeze(0).shape))
        elif self.pading == 'repeat':
          mfcc_output = repeat_padding(mfcc_output, self.feat_len)
          # print("new: {}".format(feat_mat.shape))
        else:
          raise ValueError('Padding should be zero or repeat!')

      mfcc_output = mfcc_output.unsqueeze(1).float()

    return mfcc_output


#################
## LFCC front-end
#################

class LFCC(nn.Module):
  """ Based on asvspoof.org baseline Matlab code.
  Difference: with_energy is added to set the first dimension as energy
  """
  def __init__(self, batch_size, fl, fs, fn, sr, filter_num, feat_len, pading = 'repeat', 
         with_energy=True, with_emphasis=True,
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
    # print("fb_feature:{}".format(fb_feature.size()))
    # DCT (if necessary, remove DCT)
    lfcc = self.l_dct(fb_feature) if not self.flag_for_LFB else fb_feature
    
    # Add energy 
    if self.with_energy:
      power_spec = sp_amp / self.fn
      energy = torch.log10(power_spec.sum(axis=2)+ 
                 torch.finfo(torch.float32).eps)
      lfcc[:, :, 0] = energy
    # print("lfcc.size(): {}".format(lfcc.size()))

    # Add delta coefficients
    if self.with_delta:
      lfcc_delta = delta(lfcc)
      lfcc_delta_delta = delta(lfcc_delta)
      lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
      lfcc_output = lfcc_output.permute(0, 2, 1).contiguous()
      this_feat_len = lfcc_output.shape[2]           
      # print("lfcc_output.size(): {}".format(lfcc_output.size()))
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

    else:
      lfcc_output = lfcc
      lfcc_output = lfcc_output.permute(0, 2, 1).contiguous()
      this_feat_len = lfcc_output.shape[2]           

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

    return lfcc_output

#----------------------------------Sinc-conv code from Sincnet model

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

#------------------------Sinc-conv code over-------------------------------------------------
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

class LFCC_model(Model):
    def __init__(self, num_nodes, enc_dim, batch_size, frame_length, \
        frame_shift,fft_points, sr, filter_num, feat_len, resnet_type='18', nclasses=2, pad_type = 'zero', device = 'cuda', cmvn = False, frontend_name = 'LFCC_model'):
        self.in_planes = 16
        super(LFCC_model, self).__init__(frontend_name)

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fft_points = fft_points
        self.sr = sr
        self.filter_num = filter_num
        self.feat_len = feat_len
        self.pad_type = pad_type
        self.sinc = sinc
        self.cmvn = cmvn 
        # print("-------------------",self.cmvn)
        self.frontend = frontend_name
        self.front_end_LFCC = LFCC(self.batch_size,self.frame_length,self.frame_shift,self.fft_points,self.sr,self.filter_num,self.feat_len,self.pad_type, device = 'cuda')
        self.front_end_Sincnet = SincConv_fast(60, 251, 16000, pading = self.pad_type)
        self.CMVN = T.SlidingWindowCmn(cmn_window = 600, norm_vars = True)
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
        if self.frontend == 'LFCC':
            # print("---------------hi")
            x = self.front_end_LFCC(x)
        elif self.frontend == 'SincNet':
            x = x.unsqueeze(1)
            x = self.front_end_Sincnet(x)
        if self.cmvn == True:
            x = self.CMVN(x)
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

class compile_model(Model):
    def __init__(self, num_nodes, enc_dim, batch_size, frame_length, \
        frame_shift,fft_points, sr, filter_num, feat_len, resnet_type='18', nclasses=2, pad_type = 'zero', device = 'cuda', cmvn = False, frontend_name = 'LFCC_model'):
        self.in_planes = 16
        super(compile_model, self).__init__(frontend_name)

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fft_points = fft_points
        self.sr = sr
        self.filter_num = filter_num
        self.feat_len = feat_len
        self.pad_type = pad_type
        self.sinc = sinc
        self.cmvn = cmvn 
        # print("-------------------",self.cmvn)
        self.frontend = frontend_name
        # self.frontend_dict = {
        #     'LFCC': LFCC(self.batch_size,self.frame_length,self.frame_shift,self.fft_points,self.sr,self.filter_num,self.feat_len,self.pad_type,device = 'cuda'),
        #     'SincNet': SincConv_fast(60, 251, 16000, pading = self.pad_type),
            # 'LFB' : LFCC(self.batch_size,self.frame_length,self.frame_shift,self.fft_points,self.sr,self.filter_num,self.feat_len,self.pad_type,flag_for_LFB=True,device = 'cuda'),#LFB(self.batch_size,self.frame_length,self.frame_shift,self.fft_points,self.sr,self.filter_num,self.feat_len,self.pad_type),
            # 'Spectrogram': Spectrogram(self.frame_length, self.frame_shift, self.fft_points, self.sr, self.feat_len, self.pad_type),
            # 'MFCC': MFCC(self.frame_length, self.frame_shift, self.fft_points, self.sr, self.filter_num, self.feat_len, self.pad_type),
        #     }
        self.frontend_LFCC = LFCC(self.batch_size,self.frame_length,self.frame_shift,self.fft_points,self.sr,self.filter_num,self.feat_len,self.pad_type, device = 'cuda')
        self.frontend_LFB = LFCC(self.batch_size,self.frame_length,self.frame_shift,self.fft_points,self.sr,self.filter_num,self.feat_len,self.pad_type,flag_for_LFB=True,device = 'cuda')#LFB(self.batch_size,self.frame_length,self.frame_shift,self.fft_points,self.sr,self.filter_num,self.feat_len,self.pad_type),
        self.frontend_Spectrogram = MelSpec2DDCT(self.frame_length, self.frame_shift, self.fft_points, self.sr, self.filter_num, self.feat_len, self.pad_type, device = 'cuda')
        self.frontend_MFCC = MFCC(self.batch_size,self.frame_length, self.frame_shift, self.fft_points, self.sr, self.filter_num, self.feat_len, self.pad_type, device = 'cuda')
        self.frontend_Sincnet = SincConv_fast(60, 251, 16000, pading = self.pad_type)
        self.CMVN = T.SlidingWindowCmn(cmn_window = 600, norm_vars = True)
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
        if self.frontend == 'LFCC':
            x = self.frontend_LFCC(x)
        elif self.frontend == 'SincNet':
            x = x.unsqueeze(1)
            x = self.frontend_Sincnet(x)
        elif self.frontend == 'LFB':
            x = self.frontend_LFB(x)
        elif self.frontend == 'Spectrogram':
            x = self.frontend_Spectrogram(x)
        elif self.frontend == 'MFCC':
            x = self.frontend_MFCC(x)
        # if (self.frontend == 'LFCC'):
        #     print("Hiiiiiiiiiiii")
        #     x = self.frontend_dict[self.frontend](x)
        #     #   x = self.front_end_LFCC(x)
        # elif (self.frontend == 'SincNet'):
        #     x = x.unsqueeze(1)
        #     # x = self.front_end_Sincnet(x)
        #     x = self.frontend_dict[self.frontend](x)
        # elif (self.frontend == 'MFCC'):
        #     x = self.frontend_dict[self.frontend](x)
        # elif (self.frontend == 'LFB'):
        #     # print("Running LFB")
        #     x = self.frontend_dict[self.frontend](x)
        # elif(self.frontend == 'Spectrogram'):
        #     x = self.frontend_dict[self.frontend](x)
        if self.cmvn == True:
            x = self.CMVN(x)
        x = self.conv1(x)
        x = self.act(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        print("----------x.size: {}".format(x.size()))
        x = self.act(self.bn5(x)).squeeze(2)
        print("----------x.size: {}".format(x.size()))
        x = x.permute(0, 2, 1).contiguous()
        # print("----------x.size: {}".format(x.size()))
        stats = self.attention(x)

        feat = self.fc(stats)

        mu = self.fc_mu(feat)

        return feat, mu
#------------------------------Code from OC OVER-------------------------------------------
