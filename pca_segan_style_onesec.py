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
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA


# trying to create CM score for eval dataset for segan + OC model ie mixture of clean.py from segan and test.py from OC model.
# change protocol file for eval set and replace the name of the file with their respective the absolute paths.

# this file makes us upload train config seperatly, so we can upload various train configs according to our needs
#(applicable for various experiments done with SEGAN+OC(LFCC) not for other versions of the model)(SEGAN+OC(LFCC), SEGAN+OC(Sincent), SEGAN+OC(LFCC+Sincent))
# that is, if we want CM scores from segan + OC(ocsoftmax) we will have to upload the config file of that trained model
# and also don't forget to enter arguments for this file correctly based on the args of the training config
class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

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
    segan = OC_model(args)
    if opts.oc_pretrained_ckpt is not None:   
        segan.OC.load_pretrained(opts.oc_pretrained_ckpt,True)
        if args.add_loss_oc == "amsoftmax" and opts.ams_pretrained_ckpt is not None:         
            segan.AMS.load_pretrained(opts.ams_pretrained_ckpt,True)
        if args.add_loss_oc == "ocsoftmax" and opts.ocs_pretrained_ckpt is not None:
            segan.OCS.load_pretrained(opts.ocs_pretrained_ckpt,True)
        if (args.add_loss_oc == "MV-AM" or args.add_loss_oc == "MV-Arc") and opts.mvs_pretrained_ckpt is not None:
            segan.MVS.load_pretrained(opts.mvs_pretrained_ckpt, True)

    if opts.cuda:
        segan.cuda()
    if args.oc_model:
        segan.OC.eval()
    if args.add_loss_oc == "amsoftmax":
        segan.AMS.eval()
    if args.add_loss_oc == "ocsoftmax":
        segan.OCS.eval()
    if args.add_loss_oc == "MV-AM" or args.add_loss_oc == "MV-Arc":
        segan.MVS.eval()
    
    if opts.protocol_address_dev != None:
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

    elif opts.protocol_address_eval != None:
        filehandle = open(opts.protocol_address_eval, 'r')
        protocol = []
        while True:
            # read a single line

            line = (filehandle.readline())

            protocol.append(line)

            if not line:
                break


            # close the pointer to that file
        filehandle.close()

        protocol = [s[:-1] for s in protocol]

        

        protocol = pd.DataFrame(protocol)

        protocol.columns = ['clean_abs']

        # protocol = protocol[['speaker_id', 'clean_abs', 'system_id', 'label', 'noisy_abs', 'degradation_type']]

        protocol.dropna(inplace=True)

        protocol.drop_duplicates(subset="clean_abs", keep='first', inplace=True)
        print(protocol.head())

    j = 0
    embeddings = torch.zeros((0, args.enc_dim_oc), dtype=torch.float32)
    test_targets = []
    for i in tqdm(range(len(protocol))):
        if opts.protocol_address_eval != None:
            file_id = protocol.iloc[i, 0]
            name = protocol.iloc[i,0].split('/')[-1]
        else:    
            file_id = protocol.iloc[i, 4]
            # print(file_id)
            degradation_type = protocol.iloc[i,5]
            name = protocol.iloc[i,1].split('/')[-1]
        
        [signal, fs] = sf.read(file_id)
        signal = pre_emphasize(signal, args.preemph)
        # signal = normalize_wave_minmax(signal) #this line makes audio inaudioble but i think it is present in the dataloader so it might improve result of the model

        beg_samp = 0
        end_samp = args.slice_size
        wshift = int(args.data_stride * args.slice_size)
        wlen = int(args.slice_size)
        # Batch_dev = args.batch_size
        N_fr = max(math.ceil((signal.shape[0] - wlen) / (wshift)), 0)
        # print("--------------N_fr:{}".format(N_fr+1))
        
        signal = torch.from_numpy(signal)
        sig_arr = torch.zeros([N_fr + 1, wlen])
        pout = torch.zeros(N_fr + 1)
        
        signal = signal.cuda().float().contiguous()
        sig_arr = sig_arr.cuda().float().contiguous()
        pout = pout.cuda().float().contiguous()
        count_fr = 0
        # print("Name: {} siganl.shape: {}".format(name,signal.shape))
        
        while end_samp < signal.shape[0]:
            # print("------------beg_sample : {}, end_sample: {}".format(beg_samp,end_samp))
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp + wshift
            end_samp = beg_samp + wlen
            # print("------------beg_sample : {}, end_sample: {}".format(beg_samp,end_samp))
            count_fr = count_fr + 1
        if end_samp > signal.shape[0]:
            # print("count_fr: {}".format(count_fr))
            # print("sig_arr.shape:{}".format(sig_arr.shape))
            # print("-------------Name: {} signal.shape: {}".format(name,signal.shape)) 
            # print("-------------signal[beg_samp:signal.shape[0]].shape[0]:{}".format(signal[beg_samp:signal.shape[0]].shape[0]))
            # print("----------------------(signal.shape[0]- beg_samp):{}".format((signal.shape[0] - beg_samp)))
            # assert signal[beg_samp:signal.shape[0]].shape[0] == (signal.shape[0] - beg_samp), print("damn")
            sig_arr[count_fr, :(signal.shape[0] - beg_samp)] = signal[beg_samp:signal.shape[0]]
        # print("sig_arr.shape:{}".format(sig_arr.shape))
        
            

        if count_fr >= 0:
            j = j + 1
            # print(j)

            inp = sig_arr
            # print("------inp.size out", inp.shape)

            feats, lfcc_outputs = segan.OC(inp)
            embeddings = torch.cat((embeddings, feats.detach().cpu()), 0)
            if opts.protocol_address_dev != None:
                lab_batch = spoof_genuine[protocol.iloc[i, 3]]
                labels = [lab_batch for s in range(feats.size(0))] 
            else:
                labels = [0 for s in range(feats.size(0))]
            test_targets.extend(lables.detach().cpu().tolist())
            

        
    embeddings = np.array(test_embeddings)
    test_targets = np.array(test_targets) 

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

    print(j)


if __name__ == '__main__':
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
    parser.add_argument('--soundfile', action='store_true', default=False)
    parser.add_argument('--cfg_file', type=str, default=None)
    parser.add_argument('--output_folder', type = str, default = None)
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
