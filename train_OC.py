import argparse
import os
import json
import shutil
# from resnet import setup_seed, ResNet
from segan.datasets import SEDataset, SEH5Dataset, collate_fn
from segan.models.loss import *
# from dataset import ASVspoof2019
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from segan.models.lfcc_utilities import *

torch.set_default_tensor_type(torch.FloatTensor)

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

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data folder prepare
    # parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    # parser.add_argument("-f", "--path_to_features", type=str, help="features path",
    #                     default='/dataNVME/neil/ASVspoof2019LAFeatures/')
    # parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
    #                     default='/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=50, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    # parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=598)

    parser.add_argument('--add_loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax")

    parser.add_argument('--continue_training', action='store_true', help="continue training with previously trained model")

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


    #------------------------AGS to help the use of SEGAN dataloader with this model-------------------
    parser.add_argument('--protocol_address', type=str, 
                        default = 'dataset/protocol')
    parser.add_argument('--protocol_address_val', type=str, 
                        default = None)
    parser.add_argument('--slice_size', type=int, default=16384)
    parser.add_argument('--data_stride', type=float,
                        default=0.5, help='Stride in seconds for data read')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--slice_workers', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader number of workers (Def: 1).')
    parser.add_argument('--preemph_norm', action='store_true', default=False,
                        help='Inverts old  norm + preemph order in data ' \
                        'loading, so denorm has to respect this aswell')
    parser.add_argument('--random_scale', type=float, nargs='+', 
                        default=[1], help='Apply randomly a scaling factor' \
                                          'in list to the (clean, noisy) pair')
    parser.add_argument('--cache_dir', type=str, default='data_cache')
    parser.add_argument('--preemph', type=float, default=0.95,
                        help='Wav preemphasis factor (Def: 0.95).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA even if device is available')
    parser.add_argument('--sinc', action='store_true', default=False,
                        help='If false the LFCC will be front end other wise sincconv')
    


    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Set seeds
    setup_seed(args.seed)
    
    if args.sinc == True:
        args.out_fold = args.out_fold+'/sinc'
    else: 
        args.out_fold = args.out_fold+ '/LFCC'
        # Path for output data
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
        os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

    # Path for input data
    # assert os.path.exists(args.path_to_features)

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")
    

    # assign device
    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    device = 'cpu'
    if torch.cuda.is_available and not args.no_cuda:
        device = 'cuda'
        args.cuda = True
    CUDA = (device == 'cuda')
    torch.set_default_tensor_type(torch.FloatTensor)
    prev_val_eer = 1e8
    best_val_eer = 1e8
    best_train_eer = 1e8
    prev_train_eer = 1e8
    #-----------------------in current dev set the dataloader extracts random 1 second samples and pushes it to the model which might not tell the 
#----------------------- exact performance of the model and its better to implement the validation part as it is done in the test_OC code as it will 
#-----------------------help the model jusge better
    filehandle = open(args.protocol_address_val, 'r')
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
    

    protocol = pd.DataFrame([s.strip().split(' ') for s in protocol])

    protocol.columns = ['speaker_id', 'clean_abs', 'blah', 'system_id', 'label', 'noisy_abs', 'degradation_type']

    protocol = protocol[['speaker_id', 'clean_abs', 'system_id', 'label', 'noisy_abs', 'degradation_type']]

    protocol.dropna(inplace=True)

    protocol.drop_duplicates(subset="clean_abs", keep='first', inplace=True)

    # initialize model
    # lfcc_model = ResNet(3, args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
    lfcc_model = LFCC_model(3, args.enc_dim_oc, args.batch_size, args.frame_length_oc, \
            args.frame_shift_oc,args.fft_points_oc, args.sr_oc, args.filter_num_oc, args.feat_len_oc, \
            resnet_type='18', nclasses=2, pad_type = args.pad_type_oc, sinc= args.sinc).to(args.device)
    if args.continue_training:
        lfcc_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt')).to(args.device)

    lfcc_optimizer = torch.optim.Adam(lfcc_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    # training_set = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'train',
    #                             'LFCC', feat_len=args.feat_len, padding=args.padding)
    # validation_set = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'dev',
    #                               'LFCC', feat_len=args.feat_len, padding=args.padding)
    # trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                              collate_fn=training_set.collate_fn)
    # valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                            collate_fn=validation_set.collate_fn)
    dset = SEDataset(args.protocol_address, 
                         args.preemph,
                         do_cache=True,
                         cache_dir=args.cache_dir,
                         split='train',
                         stride=args.data_stride,
                         slice_size=args.slice_size,
                         max_samples=args.max_samples,
                         verbose=True,
                         slice_workers=args.slice_workers,
                         preemph_norm=args.preemph_norm,
                         random_scale=args.random_scale
                        )
    n_classes = 2
    n_samples = int(args.batch_size/n_classes)
    balanced_batch_sampler = BalancedBatchSampler(dset, n_classes, n_samples)

    dloader = DataLoader(dset, #batch_size=args.batch_size,
                         #shuffle=True, 
                         num_workers=args.num_workers,
                         pin_memory=CUDA,
                         collate_fn=collate_fn, batch_sampler = balanced_batch_sampler)

    va_dset = SEDataset(args.protocol_address_val, 
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
                            collate_fn=collate_fn)

    # feat, _, _, _ = training_set[29]
    # print("Feature shape", feat.shape)

    criterion = nn.CrossEntropyLoss()

    if args.add_loss == "amsoftmax":
        amsoftmax_loss = AMSoftmax(2, args.enc_dim, s=args.alpha, m=args.r_real).to(args.device)
        amsoftmax_loss.train()
        amsoftmax_optimzer = torch.optim.SGD(amsoftmax_loss.parameters(), lr=0.01)

    if args.add_loss == "ocsoftmax":
        ocsoftmax = OCSoftmax(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(args.device)
        ocsoftmax.train()
        ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)

    early_stop_cnt = 0
    prev_eer = 1e8

    monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        lfcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, lfcc_optimizer, epoch_num)
        if args.add_loss == "ocsoftmax":
            adjust_learning_rate(args, ocsoftmax_optimzer, epoch_num)
        elif args.add_loss == "amsoftmax":
            adjust_learning_rate(args, amsoftmax_optimzer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        idx_loader, score_loader = [], []

        for i, (uttname, clean, noisy, slice_idx, system_id, labels, tags) in enumerate(tqdm(dloader)):

            # lfcc = lfcc.unsqueeze(1).float().to(args.device)
            clean = clean.to(args.device)
            labels = labels.to(args.device)
            feats, lfcc_outputs = lfcc_model(clean)
            lfcc_loss = criterion(lfcc_outputs, labels)
            score = F.softmax(lfcc_outputs, dim=1)[:, 0]

            if args.add_loss == "softmax":
                lfcc_optimizer.zero_grad()
                trainlossDict[args.add_loss].append(lfcc_loss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()

            if args.add_loss == "ocsoftmax":
                ocsoftmaxloss, score = ocsoftmax(feats, labels)
                lfcc_loss = ocsoftmaxloss * args.weight_loss
                lfcc_optimizer.zero_grad()
                ocsoftmax_optimzer.zero_grad()
                trainlossDict[args.add_loss].append(ocsoftmaxloss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()
                ocsoftmax_optimzer.step()

            if args.add_loss == "amsoftmax":
                outputs, moutputs = amsoftmax_loss(feats, labels)
                lfcc_loss = criterion(moutputs, labels)
                trainlossDict[args.add_loss].append(lfcc_loss.item())
                lfcc_optimizer.zero_grad()
                amsoftmax_optimzer.zero_grad()
                lfcc_loss.backward()
                lfcc_optimizer.step()
                amsoftmax_optimzer.step()
                score = F.softmax(lfcc_outputs, dim=1)[:, 0]
            idx_loader.append(labels)
            score_loader.append(score)

            
            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(np.nanmean(trainlossDict[monitor_loss])) + "\n")
        
        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        array_sum = np.sum(scores)
        array_has_nan = np.isnan(array_sum)
        train_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
        other_train_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
        train_eer = min(train_eer, other_train_eer)
        print('train_eer: {}, , scores contains None: {}'.format(train_eer, array_has_nan))
        if train_eer < best_train_eer:
                print('best train_EER: {}  Prev train_EER: {} Prev best train_EER: {}'.format(train_eer, prev_train_eer, best_train_eer))
                best_train_eer = train_eer
                prev_train_eer = train_eer
        else:
            print('Current train_EER: {}  Prev train_EER: {} Best train_EER: {}'.format(train_eer, prev_train_eer, best_train_eer))
            train_prev_eer = train_eer   
        
        # Val the model
        lfcc_model.eval()
        with torch.no_grad():
            # idx_loader, score_loader = [], []
            # for i, (uttname, clean, noisy, slice_idx, system_id, labels, tags) in enumerate(tqdm(va_dloader)):
            #     # lfcc = lfcc.unsqueeze(1).float().to(args.device)
            #     clean = clean.to(args.device)
            #     labels = labels.to(args.device)

            #     feats, lfcc_outputs = lfcc_model(clean)

            #     lfcc_loss = criterion(lfcc_outputs, labels)
            #     score = F.softmax(lfcc_outputs, dim=1)[:, 0]

            #     if args.add_loss == "softmax":
            #         devlossDict["softmax"].append(lfcc_loss.item())
            #     elif args.add_loss == "amsoftmax":
            #         outputs, moutputs = amsoftmax_loss(feats, labels)
            #         lfcc_loss = criterion(moutputs, labels)
            #         score = F.softmax(outputs, dim=1)[:, 0]
            #         devlossDict[args.add_loss].append(lfcc_loss.item())
            #     elif args.add_loss == "ocsoftmax":
            #         ocsoftmaxloss, score = ocsoftmax(feats, labels)
            #         devlossDict[args.add_loss].append(ocsoftmaxloss.item())
            #     idx_loader.append(labels)
            #     score_loader.append(score)

            # scores = torch.cat(score_loader, 0).data.cpu().numpy()
            # labels = torch.cat(idx_loader, 0).data.cpu().numpy()
#-------------------implementing proper dev methodology---------------
            idx_loader, score_loader = np.array([]), np.array([]) 
            spoof_genuine = {'spoof': 1, 'bonafide':0}
            for i in tqdm(range(len(protocol))):
                
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
                    # j = j + 1
                    # # print(j)

                    inp = sig_arr
                    # print("------inp.size out", inp.shape)

                    feats, lfcc_outputs = lfcc_model(inp)
                    
                    lab_batch = spoof_genuine[protocol.iloc[i, 3]]
                    labels = [lab_batch for s in range(feats.size(0))] 
                    

                    score = F.softmax(lfcc_outputs)[:, 0]

                    if args.add_loss == "ocsoftmax":
                        ang_isoloss, score = ocsoftmax(feats, labels)
                    elif args.add_loss == "amsoftmax":
                        outputs, moutputs = amsoftmax_loss(feats, labels)
                        score = F.softmax(outputs, dim=1)[:, 0]
            
                pout[0:] = score
                score_oc_oc = torch.sum((pout[:]), dim=0) / len(pout)
                idx_loader = np.append(idx_loader ,lab_batch)
                score_loader = np.append(score_loader ,score_oc_oc.cpu())
            scores = score_loader
            labels = idx_loader
            array_sum = np.sum(scores)
            array_has_nan = np.isnan(array_sum)
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_val_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            val_eer = min(val_eer, other_val_eer)

            
            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer) +"\n")
            
            print("Val EER: {}".format(val_eer))



        
        torch.save(lfcc_model, os.path.join(args.out_fold, 'checkpoint',
                                            'anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))
        if args.add_loss == "ocsoftmax":
            loss_model = ocsoftmax
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        elif args.add_loss == "amsoftmax":
            loss_model = amsoftmax_loss
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        else:
            loss_model = None
        

        if val_eer < best_val_eer:
            # Save the model checkpoint
            
            print('best val_EER: {}  Prev val_EER: {} Prev best val_EER: {}'.format(val_eer, prev_val_eer, best_val_eer))
            best_val_eer = val_eer
            prev_val_eer = val_eer
              
            
            torch.save(lfcc_model, os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
            if args.add_loss == "ocsoftmax":
                loss_model = ocsoftmax
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            elif args.add_loss == "amsoftmax":
                loss_model = amsoftmax_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            else:
                loss_model = None

            
            prev_eer = val_eer
            early_stop_cnt = 0
        else:

            print('Current val_EER: {}  Prev val_EER: {} Best val_EER: {}'.format(val_eer, prev_val_eer, best_val_eer))
            val_prev_eer = val_eer 
            early_stop_cnt += 1

        

        if early_stop_cnt == 100:
            
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
            
            break

    return lfcc_model, loss_model


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)
    
    model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
    if args.add_loss == "softmax":
        loss_model = None
    else:
        loss_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
    

