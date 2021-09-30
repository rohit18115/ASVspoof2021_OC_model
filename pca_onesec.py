import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# from dataset import ASVspoof2019
from oc.models import *
# from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import pandas as pd
import soundfile as sf
# from oc.datasets import *
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA



def test_model(args, protocol, output_folder, feat_model_path, loss_model_path, part, add_loss, device):



    model = torch.load(feat_model_path, map_location="cuda")
    
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None
    spoof_genuine = {'spoof': 1, 'bonafide':0}
    model.eval()
    j = 0
    embeddings = torch.zeros((0, args.enc_dim_oc), dtype=torch.float32)
    test_targets = []
    for i in tqdm(range(len(protocol))):
        if args.protocol_address_eval != None:
            file_id = protocol.iloc[i, 0]
            name = protocol.iloc[i,0].split('/')[-1]
        elif args.protocol_address_dev != None:    
            file_id = protocol.iloc[i, 4]
            name = protocol.iloc[i,1].split('/')[-1]
            degradation_type = protocol.iloc[i,5]
        elif args.protocol_address_eval_2019 != None:
            file_id = protocol.iloc[i, 1]
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
        
        signal = signal.to(device).float().contiguous()
        sig_arr = sig_arr.to(device).float().contiguous()
        pout = pout.to(device).float().contiguous()
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

            feats, lfcc_outputs = model(inp)
            embeddings = torch.cat((embeddings, feats.detach().cpu()), 0)

            if args.protocol_address_dev != None or args.protocol_address_eval_2019 != None:
                lab_batch = spoof_genuine[protocol.iloc[i, 3]]
                labels = [lab_batch for s in range(feats.size(0))] 
            elif args.protocol_address_eval != None:
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


        
        
    

def test(protocol,args):
    model_path = os.path.join(args.model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(args.model_dir, "anti-spoofing_loss_model.pt")
    test_model(args, protocol, args.output_folder, model_path, loss_model_path, args.protocol_address_dev, args.loss, args.device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/ocsoftmax")
    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--slice_size', type=int, default=16384)
    parser.add_argument('--data_stride', type=float,
                        default=0.5, help='Stride in seconds for data read')
    parser.add_argument('--oc_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--ocs_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--ams_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--protocol_address_eval', type=str, default=None)
    parser.add_argument('--protocol_address_dev', type=str, default=None)
    parser.add_argument('--protocol_address_eval_2019', type=str, default=None)
    parser.add_argument('--output_folder', type = str, default = None)
    parser.add_argument('--preemph', type = float, default = 0.95)
    parser.add_argument('--enc_dim_oc', type = int , default = 256)
    parser.add_argument('--batch_size', type = int , default = 100)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.protocol_address_dev != None:
        filehandle = open(args.protocol_address_dev, 'r')
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

    elif args.protocol_address_eval != None:
        filehandle = open(args.protocol_address_eval, 'r')
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
    if args.protocol_address_eval_2019 != None:
        filehandle = open(args.protocol_address_eval_2019, 'r')
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

        # protocol = [s.strip() for s in protocol]

        tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        spoof_genuine = {'spoof': 1, 'bonafide':0}

        protocol = pd.DataFrame([s.strip().split(' ') for s in protocol])

        protocol.columns = ['speaker_id', 'clean_abs', 'blah', 'system_id', 'label']

        protocol = protocol[['speaker_id', 'clean_abs', 'system_id', 'label']]

        protocol.dropna(inplace=True)

        protocol.drop_duplicates(subset="clean_abs", keep='first', inplace=True)
        print(protocol.head())
    test(protocol, args)
    # eer_cm_lst, min_tDCF_lst = test_individual_attacks(os.path.join(args.model_dir, 'checkpoint_cm_score.txt'))
    # print(eer_cm_lst)
    # print(min_tDCF_lst)
