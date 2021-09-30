import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from oc.models import *
from oc.datasets import *
import soundfile as sf
from scipy.io import wavfile
from torch.autograd import Variable
import numpy as np
import random
import librosa
import timeit
import json
import glob
import os
from oc.models import OC_model
from oc.datasets import SEDataset, max_collate_fn, max_collate_fn_eval, mean_collate_fn_eval, mean_collate_fn
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
def get_key(my_dict, val):
    for key, value in my_dict.items():
         if val == value:
             return key

def main(opts):
    assert opts.cfg_file is not None
    # assert opts.test_files is not None
    assert opts.oc_pretrained_ckpt is not None

    with open(opts.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        print('Loaded train config: ')
        print(json.dumps(vars(args), indent=2))
    args.cuda = opts.cuda
    # if hasattr(opts, 'oc_model') and opts.oc_model: 
    ocmodel = OC_model(args)
    if opts.oc_pretrained_ckpt is not None:   
        ocmodel.OC.load_pretrained(opts.oc_pretrained_ckpt,True)
        if args.add_loss_oc == "amsoftmax" and opts.ams_pretrained_ckpt is not None:         
            ocmodel.AMS.load_pretrained(opts.ams_pretrained_ckpt,True)
        if args.add_loss_oc == "ocsoftmax" and opts.ocs_pretrained_ckpt is not None:
            ocmodel.OCS.load_pretrained(opts.ocs_pretrained_ckpt,True)
        if (args.add_loss_oc == "MV-AM" or args.add_loss_oc == "MV-Arc") and opts.mvs_pretrained_ckpt is not None:
            ocmodel.MVS.load_pretrained(opts.mvs_pretrained_ckpt, True)
    device = 'cuda'
    if opts.cuda:
        CUDA = (device == 'cuda')
        ocmodel.cuda()
    if args.oc_model:
        ocmodel.OC.eval()
    if args.add_loss_oc == "amsoftmax":
        ocmodel.AMS.eval()
    if args.add_loss_oc == "ocsoftmax":
        ocmodel.OCS.eval()
    if args.add_loss_oc == "MV-AM" or args.add_loss_oc == "MV-Arc":
        ocmodel.MVS.eval()

    

    
    if opts.protocol_address_dev != None:
        if opts.mean_size:
            va_dset = SEDataset(opts.protocol_address_dev, 
                                args.preemph,
                                do_cache=True,
                                cache_dir=args.cache_dir,
                                split='valid',
                                stride=args.data_stride,
                                slice_size=args.slice_size,
                                max_samples=opts.max_samples,
                                verbose=True,
                                slice_workers=args.slice_workers,
                                preemph_norm=args.preemph_norm)
            va_dloader = DataLoader(va_dset, batch_size=50,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=CUDA,
                                collate_fn=mean_collate_fn)    
        else:
            va_dset = SEDataset(opts.protocol_address_dev, 
                                    args.preemph,
                                    do_cache=True,
                                    cache_dir=args.cache_dir,
                                    split='valid',
                                    stride=args.data_stride,
                                    slice_size=args.slice_size,
                                    max_samples=opts.max_samples,
                                    verbose=True,
                                    slice_workers=args.slice_workers,
                                    preemph_norm=args.preemph_norm)
            va_dloader = DataLoader(va_dset, batch_size=50,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=CUDA,
                                    collate_fn=max_collate_fn)
        filehandle = open(opts.protocol_address_dev, 'r')
        protocol = []
        while True:
            # read a single line

            line = (filehandle.readline())

            protocol.append(line)

            if not line:
                break

        # lab_batch = np.zeros(batch_size)

            # close the pointer to that file
        filehandle.close()

        protocol = [s[:-1] for s in protocol]

        tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        spoof_genuine = {'spoof': 1, 'bonafide':0}


        protocol = pd.DataFrame([s.strip().split(' ') for s in protocol])

        protocol.columns = ['speaker_id', 'clean_abs', 'blah', 'system_id', 'label', 'noisy_abs', 'degradation_type']

        protocol = protocol[['speaker_id', 'clean_abs', 'system_id', 'label', 'noisy_abs', 'degradation_type']]

        protocol.dropna(inplace=True)

        protocol.drop_duplicates(subset="clean_abs", keep='first', inplace=True)

        # self.speaker_id = protocol.iloc[:,0]
        # self.clean_names = protocol.iloc[:,1]
        # self.noisy_names = protocol.iloc[:,4]
        # self.labels = protocol.iloc[:, 3]
        degradation_types = protocol.iloc[:,5]
        # self.system_id = protocol.iloc[:,2]

        degradation_set = list(set(degradation_types))
        deg_num_list = [i for i in range(len(degradation_set))]
        deg = dict(zip(degradation_set, deg_num_list))

    elif opts.protocol_address_eval != None:
        if opts.mean_size:
            va_dset = SEDataset(opts.protocol_address_eval, 
                                args.preemph,
                                do_cache=True,
                                cache_dir=args.cache_dir,
                                split='eval',
                                stride=args.data_stride,
                                slice_size=args.slice_size,
                                max_samples=opts.max_samples,
                                verbose=True,
                                slice_workers=args.slice_workers,
                                preemph_norm=args.preemph_norm)
            va_dloader = DataLoader(va_dset, batch_size=80,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=CUDA,
                                collate_fn=mean_collate_fn_eval)    
        else:
            va_dset = SEDataset(opts.protocol_address_eval, 
                                    args.preemph,
                                    do_cache=True,
                                    cache_dir=args.cache_dir,
                                    split='eval',
                                    stride=args.data_stride,
                                    slice_size=args.slice_size,
                                    max_samples=opts.max_samples,
                                    verbose=True,
                                    slice_workers=args.slice_workers,
                                    preemph_norm=args.preemph_norm)
            va_dloader = DataLoader(va_dset, batch_size=80,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=CUDA,
                                    collate_fn=max_collate_fn_eval)

    embeddings = torch.zeros((0, args.enc_dim_oc), dtype=torch.float32)
    test_targets = []
    for bidx, batch in enumerate(tqdm(va_dloader), start=1):
        gc.collect()
        if opts.protocol_address_dev != None:
            if len(batch) == 7:
                if opts.clean:
                    uttname, signal, __, sys_id, _, labels_oc, degradation_type = batch
                else:    
                    uttname, _, signal, sys_id, _, labels_oc, degradation_type = batch
                signal = signal.cuda().float().contiguous()
                # print("----------------",type(signal))
                sys_id = sys_id.to(device)
                labels_oc = labels_oc.cuda().float().contiguous()
                degradation_type = degradation_type.to(device)
                # print("uttname: {}, label: {}, degradation_type: {}".format(uttname,labels_oc,degradation_type)) # checks for understanding code
            else:
                raise ValueError('Returned {} elements per '
                                 'sample?'.format(len(batch)))
        elif opts.protocol_address_eval != None:
            if len(batch) == 2:
                uttname, signal = batch
                signal = signal.cuda().float().contiguous()
                if args.add_loss_oc == "amsoftmax":
                    labels_oc = torch.zeros([signal.size(0)], dtype = torch.int64).to(device)
                    # print(labels_oc.size(), signal.size(0))
                elif args.add_loss_oc == "ocsoftmax":
                    labels_oc = torch.FloatTensor([0 for i in range(len(uttname))]).cuda().float().contiguous()
                # print("uttname: {}, label: {}, degradation_type: {}".format(uttname,labels_oc,degradation_type)) # checks for understanding code
            else:
                raise ValueError('Returned {} elements per '
                                 'sample?'.format(len(batch)))

        feats, lfcc_outputs = ocmodel.OC(signal)

        labels_oc = labels_oc.detach().cpu().tolist()

        embeddings = torch.cat((embeddings, feats.detach().cpu()), 0)
        test_targets.extend(labels_oc)



    embeddings = np.array(embeddings)
    test_targets = np.array(test_targets) 

    if opts.mode == 'PCA':    
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        pca_proj = pca.transform(embeddings)  
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        num_categories = 2
        for lab in range(num_categories):
            indices = test_targets==lab
            ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.savefig("output.jpg")
    elif opts.mode == 'tSNE':
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(embeddings)
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        num_categories = 2
        for lab in range(num_categories):
            indices = test_targets==lab
            ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.savefig("output.jpg")

        


        

if __name__ == '__main__':


# COMMAND TO Test the model on EVAL set, the --max_samples ensures that the dataloader does not skip the last batch due to any error or empty space in the protocol_file
#python test_OC_segan_style.py --oc_pretrained_ckpt /media/root/rohit/codes/OC_model/weights/whole_sample/OC_model/clean/random_batching/64/weights_LFCC_model-best_LFCC_model-2.ckpt --ocs_pretrained_ckpt /media/root/rohit/codes/OC_model/weights/whole_sample/OC_model/clean/random_batching/64/weights_OCS-best_OCS-2.ckpt --protocol_address_eval /media/root/rohit/datasets/LA/ASVspoof2021_LA_eval/protocol_new_eval.txt --cfg_file /media/root/rohit/codes/OC_model/weights/whole_sample/OC_model/clean/random_batching/64/Segan/train.opts --output_folder /media/root/rohit/codes/OC_model/weights/whole_sample/OC_model/clean/random_batching/64/scores/eval_2021/ --max_samples 181566

    parser = argparse.ArgumentParser()
    # parser.add_argument('--g_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--oc_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--ocs_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--ams_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--mvs_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--protocol_address_eval', type=str, default=None)
    parser.add_argument('--protocol_address_dev', type=str, default=None)
    parser.add_argument('--h5', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    # parser.add_argument('--synthesis_path', type=str, default='segan_samples',
    #                     help='Path to save output samples (Def: ' \
    #                          'segan_samples).')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--clean', action='store_true', default=True)
    parser.add_argument('--soundfile', action='store_true', default=False)
    parser.add_argument('--cfg_file', type=str, default=None)
    parser.add_argument('--output_folder', type = str, default = None)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--mean_size', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='PCA', help='PCA or tSNE')

    # parser.add_argument('--mode', type = str, default = 'SEGAN+OC(LFCC)', 
    #                     help = 'options for mode are as follows: \
    #                     SEGAN+OC(LFCC), SEGAN+OC(Sincent), SEGAN+OC(LFCC+Sincent)')

    opts = parser.parse_args()

    # if not os.path.exists(opts.synthesis_path):
    #     os.makedirs(opts.synthesis_path)
    
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)
    

    main(opts)
