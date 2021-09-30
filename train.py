import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from oc.models import OC_model
from oc.datasets import SEDataset, SEH5Dataset, collate_fn, max_collate_fn,mean_collate_fn, BalancedBatchSampler
from oc.utils import Additive
import numpy as np
import random
import json
import os

def main(opts):
    # select device to work on 
    device = 'cpu'
    if torch.cuda.is_available and not opts.no_cuda:
        device = 'cuda'
        opts.cuda = True
    CUDA = (device == 'cuda')
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if CUDA:
        torch.cuda.manual_seed_all(opts.seed)
    # create OC model
    # if opts.wsegan:
    #     segan = WSEGAN(opts)
    # elif opts.aewsegan:
    #     segan = AEWSEGAN(opts)
    # elif opts.segan_oc:
    #     segan = SEGAN_OC(opts)
    # elif opts.segan:
    #     segan = SEGAN(opts)
    if opts.oc_model:
        ocmodel = OC_model(opts)
    ocmodel.to(device)
    # possibly load pre-trained sections of networks G or D
    print('Total model parameters: ',  ocmodel.get_n_params())
    
    if hasattr(opts, 'oc_model') and opts.oc_model: 
        if opts.oc_pretrained_ckpt is not None:   
            ocmodel.OC.load_pretrained(opts.oc_pretrained_ckpt,True)
            if opts.add_loss_oc == "amsoftmax" and opts.ams_pretrained_ckpt is not None:         
                ocmodel.AMS.load_pretrained(opts.ams_pretrained_ckpt,True)
            if opts.add_loss_oc == "ocsoftmax" and opts.ocs_pretrained_ckpt is not None:
                ocmodel.OCS.load_pretrained(opts.ocs_pretrained_ckpt,True)
            if (opts.add_loss_oc == "MV-AM" or opts.add_loss_oc == "MV-Arc") and opts.mvs_pretrained_ckpt is not None:
                ocmodel.MVS.load_pretrained(opts.mvs_pretrained_ckpt,True)
    n_classes = 2
    n_samples = int(opts.batch_size/n_classes)

    # create Dataset(s) and Dataloader(s)
    if opts.h5:
        # H5 Dataset with processed speech chunks
        if opts.h5_data_root is None:
            raise ValueError('Please specify an H5 data root')
        dset = SEH5Dataset(opts.h5_data_root, split='train',
                           preemph=opts.preemph,
                           verbose=True,
                           random_scale=opts.random_scale)
    else:
        # Directory Dataset from raw wav files
        dset = SEDataset(opts.protocol_address, 
                         opts.preemph,
                         do_cache=True,
                         cache_dir=opts.cache_dir,
                         split='train',
                         stride=opts.data_stride,
                         slice_size=opts.slice_size,
                         max_samples=opts.max_samples,
                         verbose=True,
                         slice_workers=opts.slice_workers,
                         onesec=opts.onesec,
                         preemph_norm=opts.preemph_norm,
                         random_scale=opts.random_scale
                        )
    if opts.onesec == True:
        if opts.equal_class or opts.speaker_invariant or opts.degradation_invariant:
            balanced_batch_sampler = BalancedBatchSampler(opts, dset, n_classes, n_samples)
            dloader = DataLoader(dset, #batch_size=opts.batch_size,
                                 #shuffle=True, 
                                 num_workers=opts.num_workers,
                                 pin_memory=CUDA,
                                 collate_fn=collate_fn, batch_sampler = balanced_batch_sampler)
        else:
            dloader = DataLoader(dset, batch_size=opts.batch_size,
                                 shuffle=True, 
                                 num_workers=opts.num_workers,
                                 pin_memory=CUDA,
                                 collate_fn=collate_fn)
    elif opts.mean_size == True:
        if opts.equal_class or opts.speaker_invariant or opts.degradation_invariant:
            balanced_batch_sampler = BalancedBatchSampler(opts, dset, n_classes, n_samples)
            dloader = DataLoader(dset, #batch_size=opts.batch_size,
                                 #shuffle=True, 
                                 num_workers=opts.num_workers,
                                 pin_memory=CUDA,
                                 collate_fn=mean_collate_fn, batch_sampler = balanced_batch_sampler)
        else:
            dloader = DataLoader(dset, batch_size=opts.batch_size,
                                 shuffle=True, 
                                 num_workers=opts.num_workers,
                                 pin_memory=CUDA,
                                 collate_fn=mean_collate_fn)
    else:
        if opts.equal_class or opts.speaker_invariant or opts.degradation_invariant:
            balanced_batch_sampler = BalancedBatchSampler(opts, dset, n_classes, n_samples)
            dloader = DataLoader(dset, #batch_size=opts.batch_size,
                                 #shuffle=True, 
                                 num_workers=opts.num_workers,
                                 pin_memory=CUDA,
                                 collate_fn=max_collate_fn, batch_sampler = balanced_batch_sampler)
        else:
            dloader = DataLoader(dset, batch_size=opts.batch_size,
                                 shuffle=True, 
                                 num_workers=opts.num_workers,
                                 pin_memory=CUDA,
                                 collate_fn=max_collate_fn)
    if opts.protocol_address_val is not None:
        if opts.h5:
            dset = SEH5Dataset(opts.h5_data_root, split='valid',
                               preemph=opts.preemph,
                               verbose=True)
        else:
            va_dset = SEDataset(opts.protocol_address_val, 
                                opts.preemph,
                                do_cache=True,
                                cache_dir=opts.cache_dir,
                                split='valid',
                                stride=opts.data_stride,
                                slice_size=opts.slice_size,
                                max_samples=opts.max_samples,
                                verbose=True,
                                slice_workers=opts.slice_workers,
                                onesec=opts.onesec,
                                preemph_norm=opts.preemph_norm)
        if opts.onesec == True:
            va_dloader = DataLoader(va_dset, batch_size=50,
                                shuffle=False, num_workers=opts.num_workers,
                                pin_memory=CUDA,
                                collate_fn=collate_fn)    
        elif opts.mean_size ==True:
            va_dloader = DataLoader(va_dset, batch_size=50,
                                shuffle=False, num_workers=opts.num_workers,
                                pin_memory=CUDA,
                                collate_fn=mean_collate_fn)
        else:
            va_dloader = DataLoader(va_dset, batch_size=50,
                                shuffle=False, num_workers=opts.num_workers,
                                pin_memory=CUDA,
                                collate_fn=max_collate_fn)
        
    else:
        va_dloader = None
    criterion_oc = nn.CrossEntropyLoss()
    if opts.oc_model:
        ocmodel.train(opts, dloader, criterion_oc, opts.save_freq, opts.val_save_freq, device=device, va_dloader=va_dloader)  
    



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
#--------------------TRAINING CUSTOM MINI BATCH STRATEGY ARGS------------------------------------------
    
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--cmvn', action='store_true', default=False)
    parser.add_argument('--equal_class', action='store_true', default=False,
                        help='EQUAL NUMBER OF SPOOFED AND BONAFIDE SAMPLES')
    parser.add_argument('--speaker_invariant', action='store_true', default=False,
                        help='EQUAL NUMBER OF SPOOFED AND BONAFIDE SAMPLES WHERE EACH SPOOFED SAMPLE WILL HAVE A BONAFIDE SAMPLE WITH SAME SPEAKER')
    parser.add_argument('--degradation_invariant', action='store_true', default=False,
                        help='EQUAL NUMBER OF SPOOFED AND BONAFIDE SAMPLES WHERE EACH SPOOFED SAMPLE WILL HAVE A BONAFIDE SAMPLE WITH SAME DEGRADATION')
    parser.add_argument('--onesec', action='store_true', default=False)
    parser.add_argument('--mean_size', action='store_true', default=False)
    parser.add_argument('--loss_type', type=str, default=None, help="HardMining")
#-----------------------SEGAN ARGS START--------------------------------------

    parser.add_argument('--save_path', type=str, default="seganv1_ckpt",
                        help="Path to save models (Def: seganv1_ckpt).")
    parser.add_argument('--d_pretrained_ckpt', type=str, default=None,
                        help='Path to ckpt file to pre-load in training '
                             '(Def: None).')
    parser.add_argument('--g_pretrained_ckpt', type=str, default=None,
                        help='Path to ckpt file to pre-load in training '
                             '(Def: None).')
    parser.add_argument('--oc_pretrained_ckpt', type=str, default=None,
                        help='Path to ckpt file to pre-load in training '
                             '(Def: None).')
    parser.add_argument('--ocs_pretrained_ckpt', type=str, default=None,
                        help='Path to ckpt file to pre-load in training '
                             '(Def: None).')
    parser.add_argument('--ams_pretrained_ckpt', type=str, default=None,
                        help='Path to ckpt file to pre-load in training '
                             '(Def: None).')
    parser.add_argument('--cache_dir', type=str, default='data_cache')
    # parser.add_argument('--clean_trainset', type=str,
    #                     default='data/clean_trainset')
    # parser.add_argument('--noisy_trainset', type=str,
    #                     default='data/noisy_trainset')
    parser.add_argument('--protocol_address', type=str, 
                        default = 'dataset/protocol')
    parser.add_argument('--protocol_address_val', type=str, 
                        default = None)
    # parser.add_argument('--clean_valset', type=str,
    #                     default=None)#'data/clean_valset')
    # parser.add_argument('--noisy_valset', type=str,
    #                     default=None)#'data/noisy_valset')
    parser.add_argument('--h5_data_root', type=str, default=None,
                        help='H5 data root dir (Def: None). The '
                             'files will be found by split name '
                             '{train, valid, test}.h5')
    parser.add_argument('--h5', action='store_true', default=False,
                        help='Activate H5 dataset mode (Def: False).')
    parser.add_argument('--data_stride', type=float,
                        default=0.5, help='Stride in seconds for data read')
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=100,
                        help='If validation path is set, there are '
                             'denoising evaluations running for which '
                             'COVL, CSIG, CBAK, PESQ and SSNR are '
                             'computed. Patience is number of validation '
                             'epochs to wait til breakining train loop. This '
                             'is an unstable and slow process though, so we'
                             'avoid patience by setting it high atm (Def: 100).'
                       )
    
    
    parser.add_argument('--save_freq', type=int, default=50,
                        help="Batch save freq (Def: 50).")
    parser.add_argument('--val_save_freq', type=int, default=4,
                        help="Batch save freq (Def: 50).")
    parser.add_argument('--slice_size', type=int, default=16384)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--l1_dec_epoch', type=int, default=100)
    parser.add_argument('--l1_weight', type=float, default=100,
                        help='L1 regularization weight (Def. 100). ')
    parser.add_argument('--l1_dec_step', type=float, default=1e-5,
                        help='L1 regularization decay factor by batch ' \
                             '(Def: 1e-5).')
    parser.add_argument('--g_lr', type=float, default=0.00005, 
                        help='Generator learning rate (Def: 0.00005).')
    parser.add_argument('--d_lr', type=float, default=0.00005, 
                        help='Discriminator learning rate (Def: 0.0005).')
    parser.add_argument('--preemph', type=float, default=0.95,
                        help='Wav preemphasis factor (Def: 0.95).')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--eval_workers', type=int, default=5)
    parser.add_argument('--slice_workers', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader number of workers (Def: 1).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA even if device is available')
    parser.add_argument('--random_scale', type=float, nargs='+', 
                        default=[1], help='Apply randomly a scaling factor' \
                                          'in list to the (clean, noisy) pair')
    parser.add_argument('--no_train_gen', action='store_true', default=True, 
                       help='Do NOT generate wav samples during training')
    parser.add_argument('--preemph_norm', action='store_true', default=False,
                        help='Inverts old  norm + preemph order in data ' \
                        'loading, so denorm has to respect this aswell')
    parser.add_argument('--wsegan', action='store_true', default=False)
    parser.add_argument('--aewsegan', action='store_true', default=False)
    parser.add_argument('--segan_oc', action='store_true',default = False)
    parser.add_argument('--segan', action='store_true',default = False)
    parser.add_argument('--oc_model', action='store_true',default = False)
    parser.add_argument('--vanilla_gan', action='store_true', default=False)
    parser.add_argument('--no_bias', action='store_true', default=False,
                        help='Disable all biases in Generator')
    parser.add_argument('--joint_adv_training', action='store_true', default=True,
                        help='if true then discriminator will be used in training')
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--reg_loss', type=str, default='l1_loss',
                        help='Regression loss (l1_loss or mse_loss) in the '
                             'output of G (Def: l1_loss)')

    # Skip connections options for G
    parser.add_argument('--skip_merge', type=str, default='concat')
    parser.add_argument('--skip_type', type=str, default='alpha',
                        help='Type of skip connection: \n' \
                        '1) alpha: learn a vector of channels to ' \
                        ' multiply elementwise. \n' \
                        '2) conv: learn conv kernels of size 11 to ' \
                        ' learn complex responses in the shuttle.\n' \
                        '3) constant: with alpha value, set values to' \
                        ' not learnable, just fixed.\n(Def: alpha)')
    parser.add_argument('--skip_init', type=str, default='one',
                        help='Way to init skip connections (Def: one)')
    parser.add_argument('--skip_kwidth', type=int, default=11)

    # Generator parameters
    parser.add_argument('--gkwidth', type=int, default=31)
    parser.add_argument('--genc_fmaps', type=int, nargs='+',
                        default=[64, 128, 256, 512, 1024],
                        help='Number of G encoder feature maps, ' \
                             '(Def: [64, 128, 256, 512, 1024]).')
    parser.add_argument('--genc_poolings', type=int, nargs='+',
                        default=[4, 4, 4, 4, 4],
                        help='G encoder poolings')
    parser.add_argument('--z_dim', type=int, default=1024)
    parser.add_argument('--gdec_fmaps', type=int, nargs='+',
                        default=None)
    parser.add_argument('--gdec_poolings', type=int, nargs='+',
                        default=None, 
                        help='Optional dec poolings. Defaults to None '
                             'so that encoder poolings are mirrored.')
    parser.add_argument('--gdec_kwidth', type=int, 
                        default=None)
    parser.add_argument('--gnorm_type', type=str, default=None,
                        help='Normalization to be used in G. Can '
                        'be: (1) snorm, (2) bnorm or (3) none '
                        '(Def: None).')
    parser.add_argument('--no_z', action='store_true', default=False)
    parser.add_argument('--no_skip', action='store_true', default=False)
    parser.add_argument('--pow_weight', type=float, default=0.001)
    parser.add_argument('--misalign_pair', action='store_true', default=False)
    parser.add_argument('--interf_pair', action='store_true', default=False)

    # Discriminator parameters
    parser.add_argument('--denc_fmaps', type=int, nargs='+',
                        default=[64, 128, 256, 512, 1024],
                        help='Number of D encoder feature maps, ' \
                             '(Def: [64, 128, 256, 512, 1024]')
    parser.add_argument('--dpool_type', type=str, default='none',
                        help='conv/none/gmax/gavg (Def: none)')
    parser.add_argument('--dpool_slen', type=int, default=16,
                        help='Dimension of last conv D layer time axis'
                             'prior to classifier real/fake (Def: 16)')
    parser.add_argument('--dkwidth', type=int, default=None,
                        help='Disc kwidth (Def: None), None is gkwidth.')
    parser.add_argument('--denc_poolings', type=int, nargs='+', 
                        default=[4, 4, 4, 4, 4],
                        help='(Def: [4, 4, 4, 4, 4])')
    parser.add_argument('--dnorm_type', type=str, default='bnorm',
                        help='Normalization to be used in D. Can '
                        'be: (1) snorm, (2) bnorm or (3) none '
                        '(Def: bnorm).')
    parser.add_argument('--phase_shift', type=int, default=5)
    parser.add_argument('--sinc_conv', action='store_true', default=False)

#----------------------SEGAN ARGS OVER----------------------------------------------

#----------------------ARGS FOR ONE CLASS (OC) CLASSIFICATION MODEL------------------
    # Data folder prepare
    
    # parser.add_argument("-o", "--out_fold_oc", type=str, help="output folder", required=True, default='oc_classification')

    # Dataset prepare
    #--------------all of these arguments will be used in LFCC_layer refer LFCC_layer.py for reference
    parser.add_argument("--feat_len_oc", type=int, help="features length", default=750)
    parser.add_argument('--pad_type_oc', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--frame_length_oc", type=int, help="length of frame to extract LFCC", default=320)
    parser.add_argument("--enc_dim_oc",type=int, help="encoder dimension for LFCC resnet model", default=256)
    parser.add_argument("--frame_shift_oc", type=int, help="frame shift", default=160)
    parser.add_argument("--fft_points_oc", type=int, help="FFT points", default=512)
    parser.add_argument("--sr_oc", type=int, help="sampling rate", default=16000)
    parser.add_argument("--filter_num_oc", type=int, help="number of filters in the filter bank", default=20)

    # Training hyperparameters
   
    parser.add_argument('--lr_oc', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay_oc', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval_oc', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1_oc', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2_oc', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps_oc', type=float, default=1e-8, help="epsilon for Adam")
    # parser.add_argument("--gpu_oc", type=str, help="GPU index", default="1")
    # parser.add_argument('--num_workers_oc', type=int, default=0, help="number of workers")
    # parser.add_argument('--seed', type=int, help="random number seed", default=598)

    parser.add_argument('--add_loss_oc', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax', 'MV-AM', 'MV-Arc'], help="loss for one-class training")
    parser.add_argument('--weight_loss_oc', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real_oc', type=float, default=0.5, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake_oc', type=float, default=0.3, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha_oc', type=float, default=20, help="scale factor for ocsoftmax")

    parser.add_argument('--continue_training_oc', action='store_true', help="continue training with previously trained model")
    parser.add_argument('--frontend_name', default='LFCC', help='Type the frontend name')

#----------------------------------------OC ARGS SECTION OVER------------------------------

    opts = parser.parse_args()
    opts.bias = not opts.no_bias 

    if opts.segan_oc and opts.sinc:
        opts.save_path = opts.save_path+'/Segan+OC/sinc' 
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
            cfg_f.write(json.dumps(vars(opts), indent=2))

    elif opts.segan_oc and not opts.sinc:
        opts.save_path = opts.save_path+'/Segan+OC/LFCC'
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
            cfg_f.write(json.dumps(vars(opts), indent=2))

    else:
        opts.save_path = opts.save_path+'/Segan'
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
            cfg_f.write(json.dumps(vars(opts), indent=2))

#--------------------------------OC initialization-------------------------------

    # if not os.path.exists(opts.out_fold_oc):
    #     os.makedirs(opts.out_fold_oc)
    # else:
    #     shutil.rmtree(opts.out_fold_oc)
    #     os.mkdir(opts.out_fold_oc)

    # Folder for intermediate results
    # if not os.path.exists(os.path.join(opts.out_fold_oc, 'checkpoint')):
    #     os.makedirs(os.path.join(opts.out_fold_oc, 'checkpoint'))
    # else:
    #     shutil.rmtree(os.path.join(opts.out_fold_oc, 'checkpoint'))
    #     os.mkdir(os.path.join(opts.out_fold_oc, 'checkpoint'))

    # Save training arguments
    # with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
    #     file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    # with open(os.path.join(opts.out_fold_oc, 'train_loss.log'), 'w') as file:
    #     file.write("Start recording training loss ...\n")
    # with open(os.path.join(opts.out_fold_oc, 'dev_loss.log'), 'w') as file:
    #     file.write("Start recording validation loss ...\n")

    # assign device
    # args.cuda = torch.cuda.is_available()
    # print('Cuda device available: ', args.cuda)
    # args.device = torch.device("cuda" if args.cuda else "cpu")


#--------------------------------OC initialization OVER--------------------------

    # save opts
    
    print('Parsed arguments: ', json.dumps(vars(opts), indent=2))
    main(opts)
