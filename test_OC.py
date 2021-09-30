import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# from dataset import ASVspoof2019
from segan.models import *
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import pandas as pd
import soundfile as sf
from segan.datasets import *
from torch.autograd import Variable
import math



def test_model(args, protocol, output_folder, feat_model_path, loss_model_path, part, add_loss, device):



    model = torch.load(feat_model_path, map_location="cuda")
    
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None
    spoof_genuine = {'spoof': 1, 'bonafide':0}
    model.eval()
    j = 0
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
            if args.protocol_address_dev != None or args.protocol_address_eval_2019 != None:
                lab_batch = spoof_genuine[protocol.iloc[i, 3]]
                labels = [lab_batch for s in range(feats.size(0))] 
            elif args.protocol_address_eval != None:
                labels = [0 for s in range(feats.size(0))]


            score = F.softmax(lfcc_outputs)[:, 0]

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            
        pout[0:] = score
        score_oc = torch.sum((pout[:]), dim=0) / len(pout)
        
        if args.protocol_address_dev != None:
            with open(output_folder + "/dev_scores_" + "MC_OC" + ".txt", "a+") as score_file:
                if protocol.iloc[i, 3] == "bonafide":
                    score_file.write("{} - bonafide {} {}\n".format(name, score_oc, degradation_type ))
                else:
                    score_file.write(
                        "{} {} spoof {} {}\n".format(name, protocol.iloc[i, 2], 
                                                     score_oc, degradation_type))
        elif args.protocol_address_eval != None:
            with open(output_folder + "/eval_scores_" + "MC_OC" + ".txt", "a+") as score_file:
                    score_file.write("{} - {} \n".format(name, score_oc))
        elif args.protocol_address_eval_2019 != None:
            with open(output_folder + "/eval_scores_2019_" + "clean_OC" + ".txt", "a+") as score_file:
                if protocol.iloc[i, 3] == "bonafide":
                    score_file.write("{} - bonafide {} \n".format(name, score_oc))
                else:
                    score_file.write(
                        "{} {} spoof {} \n".format(name, protocol.iloc[i, 2], 
                                                     score_oc))
    print(j)
    # pout = torch.reshape(pout, (-1,))
    # pout = pout.cpu().numpy()
        
    # wavfile.write(opts.output_folder+'/' + name , int(16e3), pout)
                                           
    
    # if add_loss != "softmax":
    #     loss_model = nn.DataParallel(loss_model, list(range(torch.cuda.device_count())))
    # test_set = ASVspoof2019("LA", "/dataNVME/neil/ASVspoof2019LAFeatures/",
    #                         "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part,
    #                         "LFCC", feat_len=750, padding="repeat")
    # testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0,
    #                             collate_fn=test_set.collate_fn)
    # model.eval()

    # with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
    #     for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
    #         lfcc = lfcc.unsqueeze(1).float().to(device)
    #         tags = tags.to(device)
    #         labels = labels.to(device)

            

    #         for j in range(labels.size(0)):
    #             cm_score_file.write(
    #                 '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
    #                                       "spoof" if labels[j].data.cpu().numpy() else "bonafide",
    #                                       score[j].item()))

    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
    #                                         "/data/neil/DS_10283_3336/")
    # return eer_cm, min_tDCF

def test(protocol,args):
    model_path = os.path.join(args.model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(args.model_dir, "anti-spoofing_loss_model.pt")
    test_model(args, protocol, args.output_folder, model_path, loss_model_path, args.protocol_address_dev, args.loss, args.device)

def test_individual_attacks(cm_score_file):
    asv_score_file = os.path.join('/data/neil/DS_10283_3336',
                                  'LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(7,20):
        # Extract target, nontarget, and spoof scores from the ASV scores
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_sources == 'A%02d' % attack_idx]

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

        if eer_cm < other_eer_cm:
            # Compute t-DCF
            tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,
                                                        True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]

        else:
            tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == 'bonafide'],
                                                        other_cm_scores[cm_sources == 'A%02d' % attack_idx],
                                                        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
        eer_cm_lst.append(min(eer_cm, other_eer_cm))
        min_tDCF_lst.append(min_tDCF)

    return eer_cm_lst, min_tDCF_lst


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
