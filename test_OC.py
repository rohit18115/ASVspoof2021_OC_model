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
                                max_samples=args.max_samples,
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
                                    max_samples=args.max_samples,
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
                elif args.add_loss_oc == "MV-AM" or args.add_loss_oc == "MV-Arc":
                    labels_oc = [0 for i in range(len(uttname))] 

            else:
                raise ValueError('Returned {} elements per '
                                 'sample?'.format(len(batch)))

        feats, lfcc_outputs = ocmodel.OC(signal)

        score_oc = F.softmax(lfcc_outputs)[:, 0]

        if args.add_loss_oc == "ocsoftmax":
            ang_isoloss, score_oc = ocmodel.OCS(feats, labels_oc)
        elif args.add_loss_oc == "amsoftmax":
            outputs, moutputs = ocmodel.AMS(feats, labels_oc)
            score_oc = F.softmax(outputs, dim=1)[:, 0]
        elif args.add_loss_oc == "MV-AM" or args.add_loss_oc == "MV-Arc":
            moutputs_oc, score_oc = ocmodel.MVS(feats, torch.as_tensor(labels_oc).cuda())
            print(score_oc)

        if opts.protocol_address_dev != None:
            with open(opts.output_folder + "/dev_scores_" + "MC_OC" + ".txt", "a+") as score_file:
                for i in range(labels_oc.shape[0]):
                    if labels_oc[i] == 0:
                        score_file.write("{} - bonafide {} {}\n".format(uttname[i], score_oc[i], get_key(deg,degradation_type[i])))
                    else:
                        score_file.write(
                            "{} {} spoof {} {}\n".format(uttname[i], get_key(tag,sys_id[i]), 
                                                         score_oc[i], get_key(deg,degradation_type[i])))
        else:
            with open(opts.output_folder + "/eval_scores_" + "MC_OC" + ".txt", "a+") as score_file:
                for i in range(len(uttname)):
                    score_file.write("{} {} \n".format(uttname[i], score_oc[i]))


        # filehandle = open(opts.protocol_address_eval, 'r')
        # protocol = []
        # while True:
        #     # read a single line

        #     line = (filehandle.readline())

        #     protocol.append(line)

        #     if not line:
        #         break


        #     # close the pointer to that file
        # filehandle.close()

        # protocol = [s[:-1] for s in protocol]

        

        # protocol = pd.DataFrame(protocol)

        # protocol.columns = ['clean_abs']

        # # protocol = protocol[['speaker_id', 'clean_abs', 'system_id', 'label', 'noisy_abs', 'degradation_type']]

        # protocol.dropna(inplace=True)

        # protocol.drop_duplicates(subset="clean_abs", keep='first', inplace=True)
        # # print(protocol.head())

    # j = 0
    # for i in tqdm(range(len(protocol))):
    #     if opts.protocol_address_eval != None:
    #         file_id = protocol.iloc[i, 0]
    #         name = protocol.iloc[i,0].split('/')[-1]
    #     else:    
    #         file_id = protocol.iloc[i, 4]
    #         # print(file_id)
    #         degradation_type = protocol.iloc[i,5]
    #         name = protocol.iloc[i,1].split('/')[-1]
        
    #     [signal, fs] = sf.read(file_id)
    #     signal = pre_emphasize(signal, args.preemph)
    #     signal = normalize_wave_minmax(signal) #this line makes audio inaudioble but i think it is present in the dataloader so it might improve result of the model

    #     beg_samp = 0
    #     end_samp = args.slice_size
    #     wshift = int(args.data_stride * args.slice_size)
    #     wlen = int(args.slice_size)
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
    #         j = j + 1
    #         # print(j)

    #         inp = sig_arr
    #         # print("------inp.size out", inp.shape)

            
            
    #     pout[0:] = score
    #     score_oc = torch.sum((pout[:]), dim=0) / len(pout)
        
        
    # print(j)

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
