# Impact of Channel Variation on One-Class Learning for Spoof Detection
Code for the paper [Impact of Channel Variation on One-Class Learning for Spoof Detection](https://arxiv.org/abs/2109.14900)
We modified the original Resnet One-Class Classification model code to incoperate various mini-batching and data-feeding strategies to train the model.
This repository also contains the scripts we used to create the two versions of the datasets degraded using various codec simulations to perform multi-conditional training. The description of the datasets, mini-batching and data-feeding methdologies are described in the paper linked above.

## Here are the list of commands to use this repository to the fullest:

To train the model with random batching:

```
python train.py --save_path ./weights --protocol_address /ASVspoof2019_LA_cm_protocols/protocol_file.txt --protocol_address_val ASVspoof2019_LA_cm_protocols/val_protocol_file.txt --oc_model --batch_size 64 --frontend_name LFCC --add_loss_oc ocsoftmax 
```

To train the model with clean speech:

```
python train.py --save_path ./weights --protocol_address /ASVspoof2019_LA_cm_protocols/protocol_file.txt --protocol_address_val ASVspoof2019_LA_cm_protocols/val_protocol_file.txt --oc_model --batch_size 64 --frontend_name LFCC --add_loss_oc ocsoftmax --clean 
```

To train the model with custom mini-batching with equal number of spoofed and bonafide samples:

```
python train.py --save_path ./weights --protocol_address /ASVspoof2019_LA_cm_protocols/protocol_file.txt --protocol_address_val ASVspoof2019_LA_cm_protocols/val_protocol_file.txt --oc_model --batch_size 64 --frontend_name LFCC --add_loss_oc ocsoftmax --equal_class
```

To train the model with custom mini-batching with equal number of spoofed and bonafide samples where each spoofed sample has a bondafide sample of the same codec simulation(this was done to make the model learn features that are degradation invariant and help model learn features that are not biased by the characteristics of the codec simulations):

```
python train.py --save_path ./weights --protocol_address /ASVspoof2019_LA_cm_protocols/protocol_file.txt --protocol_address_val ASVspoof2019_LA_cm_protocols/val_protocol_file.txt --oc_model --batch_size 64 --frontend_name LFCC --add_loss_oc ocsoftmax --equal_class --degradation_invariant
```

To train the model with custom mini-batching with equal number of spoofed and bonafide samples where each spoofed sample has a bondafide sample of the same speaker(this was done to make the model learn features that are speaker invariant and help model learn features that are not biased by the characteristics of the speaker):

```
python train.py --save_path ./weights --protocol_address /ASVspoof2019_LA_cm_protocols/protocol_file.txt --protocol_address_val ASVspoof2019_LA_cm_protocols/val_protocol_file.txt --oc_model --batch_size 64 --frontend_name LFCC --add_loss_oc ocsoftmax --equal_class --speaker_invariant
```
To test the model: (creates a files with scores in it):

```
python test_OC.py --oc_pretrained_ckpt ./weights/OC-LFCC.ckpt --ocs_pretrained_ckpt ./weights/OCS-OCS.ckpt --protocol_address_dev /media/root/rohit/datasets/LA/ASVspoof2021_LA_eval/protocol_new_eval.txt --cfg_file ./weights/train.opts --output_folder ./weights/scores/ --max_samples 181566
```
To create a PCA/tSNE plot using the learned embedding of the model:

```
python pca_segan_style.py --oc_pretrained_ckpt ./weights/weights_OC-LFCC.ckpt --protocol_address_dev ./dataset/ASVspoof2019_LA_cm_protocols/dev_protocol.txt --cfg_file ./weights/train.opts --mode tSNE --max_samples 10000 --clean
```
## Other things that you might find useful in this repository:

+ Building a custom coallate function to pad/slice the samples to max/min length just before feeding it to the model([link](https://github.com/rohit18115/ASVspoof2021_OC_model/blob/main/pca_onesec.py)).
+ Building a custom batch sampler in pytorch to create custom mini-batches for training the model([link](https://github.com/rohit18115/ASVspoof2021_OC_model/blob/888bbce1d80e728b2d851184924c955019f5108d/oc/datasets/se_dataset.py#L778)).
+ Plot PCA/tSNE from learned embedding([link](https://github.com/rohit18115/ASVspoof2021_OC_model/blob/main/pca.py)).

## To-DO:
+ Add code for dataset modification.

