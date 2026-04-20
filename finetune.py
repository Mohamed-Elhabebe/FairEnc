import comet_ml
from comet_ml import Experiment

import os
import numpy as np
import random
import argparse
import time
import json
import pandas as pd
from collections import Counter

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import sys
sys.path.append('.')

from src.modules import *
from src import logger

parser = argparse.ArgumentParser(description='FairEnc Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--eval_set', default='val', type=str, help='options: val | test')
parser.add_argument('--summarized_note_file', default='', type=str)
parser.add_argument('--text_source', default='note', type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='vit-b16', type=str, help='options: vit-b16 | vit-l14')
parser.add_argument('--pretrained_weights', default='', type=str)

parser.add_argument("--save_all_best_epochs", action="store_true", default=False)

parser.add_argument('--attributes', nargs='+', type=str, default=['race', 'gender', 'ethnicity', 'language'], help='Array includes a combination of race|gender|ethnicity|language')

parser.add_argument('--adv_loss_type', default='CE', type=str, help='CE|BCE')
parser.add_argument('--adv_lambda', default=1, type=float, help='adversarial loss lambda in clip loss')

parser.add_argument('--vq_num_embeddings', default=256, type=int, help='Number of vectors in vector quantization matrix')
parser.add_argument('--commitment_cost', default=0.25, type=float, help='vector quantization commitment loss weight')
parser.add_argument('--vq_clip_lambda', default=1, type=float, help='quantized vector clip loss lambda')
parser.add_argument("--use_soft_quantization", action="store_true", default=False)
parser.add_argument("--non_isolate_encoder", action="store_false", dest='isolate_encoder')
parser.set_defaults(isolate_encoder=True)
parser.add_argument("--non_detach_codebook_for_probs", action="store_false", dest='detach_codebook_for_probs')
parser.set_defaults(detach_codebook_for_probs=True)
parser.add_argument("--vq_text_encoder_optimize", action="store_true", default=False)
parser.add_argument('--reg_lambda', default=1, type=float, help='mutual information regularization lambda in clip loss')

parser.add_argument('--d_pred_lr', default=0.05, type=float, help='demographic predictor initial learning rate')
parser.add_argument('--d_pred_weight_decay', default=6e-5, type=float, help='demographic predictor weight decay')

parser.add_argument('--train_dataset_type', default='Random_Demographics', type=str, help='Random_Demographics|Mixed_Notes|Orig_Unbiased_Notes')
parser.add_argument('--unbiased_prob', default=0.5, type=float, help='Probability of using unbiased note in mixed or original_unbiased notes dataset')

parser.add_argument('--text_contrastive_lambda', default=1, type=float, help='text contrastive loss lambda in clip loss')
parser.add_argument('--text_contrastive_temprature', default=0.07, type=float, help='text contrastive loss temprature in clip loss')

parser.add_argument('--comet_api_key', default='', type=str)
parser.add_argument('--comet_project_name', default='', type=str)
parser.add_argument('--comet_experiment_name', default='', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')

    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    comet_experiment = Experiment(
        api_key = args.comet_api_key,
        project_name = args.comet_project_name
    )
    comet_experiment.set_name(args.comet_experiment_name)
    
    comet_experiment_tags = ['Multi_Adversarial', 'One_Discriminator', 'MI_Regularization', 'Text_Contrastive', f'{args.adv_loss_type}_Adv', 'CLIP', args.model_arch, str(args.seed)] + args.attributes
    comet_experiment.add_tags(comet_experiment_tags)

    args_dict = vars(args)
    args_dict.pop('comet_api_key')
    args_dict.pop('comet_project_name')
    comet_experiment.log_parameters(args_dict)

    # the number of groups in each attribute
    groups_in_attrs = [3, 2, 2, 3]

    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}
    idx_to_attr = {0: 'race', 1: 'gender', 2: 'ethnicity', 3: 'language'}
    per_attr_idx_to_grp = {
        'race': {
            0: 'asian',
            1: 'black',
            2: 'white'
        },
        'gender': {
            0: 'female',
            1: 'male'
        },
        'ethnicity': {
            0: 'non-hispanic',
            1: 'hispanic'
        },
        'language': {
            0: 'english',
            1: 'spanish',
            2: 'other'
        }
    }

    model_arch_mapping = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(groups_in_attrs)):
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(groups_in_attrs[i])]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load(model_arch_mapping[args.model_arch], device=device, jit=False) #Must set jit=False for training

    demographic_predictor_num_logits = 0
    for attribute in args.attributes:
        demographic_predictor_num_logits += groups_in_attrs[attr_to_idx[attribute]]
    demographic_predictor = Adversary_Net(demographic_predictor_num_logits).to(device)

    if args.model_arch == 'vit-b16':
        embedding_dim = 512
    elif args.model_arch == 'vit-l14':
        embedding_dim = 768
    vector_quantizer = VectorQuantizer(args.vq_num_embeddings, embedding_dim, commitment_cost = args.commitment_cost).to(device)

    train_files = None
    test_files = None

    def seed_worker(worker_id):
        worker_seed = args.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.train_dataset_type == 'Random_Demographics':
        train_dataset = fair_vl_med_random_demographics_dataset(args.dataset_dir, preprocess, subset='Training', text_source=args.text_source, summarized_note_file=args.summarized_note_file, text_pairs = True)
    elif args.train_dataset_type == 'Mixed_Notes':
        train_dataset = fair_vl_med_mixed_notes_dataset(args.dataset_dir, preprocess, subset='Training', text_source=args.text_source, summarized_note_file=args.summarized_note_file, unbiased_prob = args.unbiased_prob, text_pairs = True)
    elif args.train_dataset_type == 'Orig_Unbiased_Notes':
        train_dataset = fair_vl_med_orig_unbiased_notes_dataset(args.dataset_dir, preprocess, subset='Training', text_source=args.text_source, summarized_note_file=args.summarized_note_file, unbiased_prob = args.unbiased_prob, text_pairs = True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)

    val_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Validation')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)

    test_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)
    
    logger.log(f'# of training samples: {train_dataset.__len__()}, # of validation samples: {val_dataset.__len__()}, # of testing samples: {test_dataset.__len__()}')
    
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(train_dataset)
    logger.log(f'group size on race in training set: {group_size_on_race}')
    logger.log(f'group size on gender in training set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in training set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(val_dataset)
    logger.log(f'group size on race in validation set: {group_size_on_race}')
    logger.log(f'group size on gender in validation set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in validation set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(test_dataset)
    logger.log(f'group size on race in test set: {group_size_on_race}')
    logger.log(f'group size on gender in test set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in test set: {group_size_on_ethnicity}')

    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

    if device == "cpu":
      model.float()
    else :
      clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.transformer.parameters(), "lr": args.lr},
        {"params": model.visual.parameters(), "lr": args.lr},
        {"params": vector_quantizer.parameters(), "lr": args.lr},
    ], lr=args.lr, betas=(0.1, 0.1), eps=1e-6,weight_decay=args.weight_decay)
    
    model_adv_losses = []
    demographic_predictor_adv_losses = []
    if args.adv_loss_type == 'CE':
        for attribute in args.attributes:
            model_adv_losses.append(nn.CrossEntropyLoss())
            demographic_predictor_adv_losses.append(nn.CrossEntropyLoss())
    elif args.adv_loss_type == 'BCE':
        for attribute in args.attributes:
            model_adv_losses.append(nn.BCEWithLogitsLoss())
            demographic_predictor_adv_losses.append(nn.BCEWithLogitsLoss())
    demographic_predictor_optimizer = optim.Adam(demographic_predictor.parameters(), lr = args.d_pred_lr, betas=(0.1, 0.1), eps=1e-6, weight_decay=args.d_pred_weight_decay)

    best_epoch = 0
    best_loss = 1000000
    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_between_group_disparity = None

    for epoch in range(args.num_epochs):
        avg_loss = 0

        avg_demographic_losses = [0] * len(args.attributes)
        avg_reverse_adv_losses = [0] * len(args.attributes)
        avg_mi_reg_losses = [0] * len(args.attributes)

        avg_quantized_clip_loss = 0
        avg_vq_loss = 0

        avg_text_contrastive_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            images, texts, texts2, label_and_attributes = batch

            images= images.to(device)
            texts = texts.to(device)
            texts2 = texts2.to(device)

            images_features = model.encode_image(images).float()
            demographic_logits = demographic_predictor(images_features.detach())
            attr_logits_start_index = 0
            total_demographic_loss = 0
            for i in range(len(args.attributes)):
                images_sensitive_attributes = label_and_attributes[:, attr_to_idx[args.attributes[i]] + 1].to(device)
                sensitive_attributes_logits = demographic_logits[images_sensitive_attributes != -1, attr_logits_start_index : attr_logits_start_index + groups_in_attrs[attr_to_idx[args.attributes[i]]]]
                attr_logits_start_index += groups_in_attrs[attr_to_idx[args.attributes[i]]]
                filtered_sensitive_attributes = images_sensitive_attributes[images_sensitive_attributes != -1]
                if args.adv_loss_type == 'CE':
                    sensitive_attributes_target = filtered_sensitive_attributes
                elif args.adv_loss_type == 'BCE':
                    sensitive_attributes_logits = torch.gather(sensitive_attributes_logits, 1, filtered_sensitive_attributes.unsqueeze(1)).squeeze(1)
                    sensitive_attributes_target = torch.ones_like(sensitive_attributes_logits, device = device)
                demographic_loss = demographic_predictor_adv_losses[i](sensitive_attributes_logits, sensitive_attributes_target)
                total_demographic_loss += demographic_loss

                avg_demographic_losses[i] += demographic_loss.item()
                comet_experiment.log_metric(f'Train_Step_Demographic_Loss_{args.attributes[i]}', demographic_loss.item(), step = epoch * len(train_dataloader) + batch_idx)
            demographic_predictor_optimizer.zero_grad()
            total_demographic_loss.backward()
            demographic_predictor_optimizer.step()

            optimizer.zero_grad()
            reverse_adv_loss = 0

            images_features = model.encode_image(images).float()
            demographic_logits = demographic_predictor(images_features)
            attr_logits_start_index = 0
            for i in range(len(args.attributes)):
                images_sensitive_attributes = label_and_attributes[:, attr_to_idx[args.attributes[i]] + 1].to(device)
                sensitive_attributes_logits = demographic_logits[images_sensitive_attributes != -1, attr_logits_start_index : attr_logits_start_index + groups_in_attrs[attr_to_idx[args.attributes[i]]]]
                attr_logits_start_index += groups_in_attrs[attr_to_idx[args.attributes[i]]]
                filtered_sensitive_attributes = images_sensitive_attributes[images_sensitive_attributes != -1]
                if args.adv_loss_type == 'CE':
                    sensitive_attributes_target = filtered_sensitive_attributes
                elif args.adv_loss_type == 'BCE':
                    sensitive_attributes_logits = torch.gather(sensitive_attributes_logits, 1, filtered_sensitive_attributes.unsqueeze(1)).squeeze(1)
                    sensitive_attributes_target = torch.zeros_like(sensitive_attributes_logits, device = device)
                attr_reverse_adv_loss = model_adv_losses[i](sensitive_attributes_logits, sensitive_attributes_target)
                reverse_adv_loss += attr_reverse_adv_loss

                avg_reverse_adv_losses[i] += attr_reverse_adv_loss.item()
                comet_experiment.log_metric(f'Train_Step_Adversarial_Loss_{args.attributes[i]}', attr_reverse_adv_loss.item(), step = epoch * len(train_dataloader) + batch_idx)

            quantized_features, vq_loss, probs = vector_quantizer(images_features, use_soft_quantization = args.use_soft_quantization, isolate_encoder = args.isolate_encoder, detach_codebook_for_probs = args.detach_codebook_for_probs)
            avg_vq_loss += vq_loss.item()
            comet_experiment.log_metric(f'Train_Step_Vector_Quantization_Loss', vq_loss.item(), step = epoch * len(train_dataloader) + batch_idx)

            mi_reg_loss = 0

            for i in range(len(args.attributes)):
                images_sensitive_attributes = label_and_attributes[:, attr_to_idx[args.attributes[i]] + 1].to(device)
                filtered_probs = probs[images_sensitive_attributes != -1, :]
                filtered_sensitive_attributes = images_sensitive_attributes[images_sensitive_attributes != -1]
                attr_mi = vq_mutual_information(filtered_probs, filtered_sensitive_attributes)
                mi_reg_loss += attr_mi

                avg_mi_reg_losses[i] += attr_mi.item()
                comet_experiment.log_metric(f'Train_Step_MI_Regularization_Loss_{args.attributes[i]}', attr_mi.item(), step = epoch * len(train_dataloader) + batch_idx)

            text_features1 = model.encode_text(texts)
            text_features2 = model.encode_text(texts2)
            text_contrastive_loss = contrastive_loss(text_features1, text_features2, temperature = args.text_contrastive_temprature)
            
            avg_text_contrastive_loss += text_contrastive_loss.item()
            comet_experiment.log_metric(f'Train_Step_Text_Contrastive_Loss', text_contrastive_loss.item(), step = epoch * len(train_dataloader) + batch_idx)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            if args.adv_loss_type == 'CE':
                total_loss -= args.adv_lambda * reverse_adv_loss
            elif args.adv_loss_type == 'BCE':
                total_loss += args.adv_lambda * reverse_adv_loss

            if args.vq_text_encoder_optimize:
                batch_text_features = model.encode_text(texts)
            else:
                batch_text_features = model.encode_text(texts).detach()
            quantized_text_sim = torch.matmul(quantized_features.float(), batch_text_features.float().T)
            sim_gt = torch.arange(len(images),dtype=torch.long,device=device)
            quantized_clip_loss = (loss_img(quantized_text_sim,sim_gt) + loss_txt(quantized_text_sim.T,sim_gt))/2

            total_loss += args.vq_clip_lambda * quantized_clip_loss
            total_loss += vq_loss

            avg_quantized_clip_loss += quantized_clip_loss.item()
            comet_experiment.log_metric(f'Train_Step_Quantizated_CLIP_Loss', quantized_clip_loss.item(), step = epoch * len(train_dataloader) + batch_idx)

            total_loss += args.reg_lambda * mi_reg_loss

            total_loss += args.text_contrastive_lambda * text_contrastive_loss

            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            avg_loss += total_loss.item()

            comet_experiment.log_metric('Train_Step_Total_Loss', total_loss.item(), step = epoch * len(train_dataloader) + batch_idx)

        avg_loss /= len(train_dataloader)

        avg_demographic_losses = [avg_demographic_loss / len(train_dataloader) for avg_demographic_loss in avg_demographic_losses]
        avg_reverse_adv_losses = [avg_reverse_adv_loss / len(train_dataloader) for avg_reverse_adv_loss in avg_reverse_adv_losses]
        avg_mi_reg_losses = [avg_mi_reg_loss / len(train_dataloader) for avg_mi_reg_loss in avg_mi_reg_losses]

        avg_vq_loss /= len(train_dataloader)
        avg_quantized_clip_loss /= len(train_dataloader)

        avg_text_contrastive_loss /= len(train_dataloader)

        comet_experiment.log_metric('Train_Total_Loss', avg_loss, step = epoch)
        for i in range(len(args.attributes)):
            comet_experiment.log_metric(f'Train_Adversarial_Loss_{args.attributes[i]}', avg_reverse_adv_losses[i], step = epoch)
            comet_experiment.log_metric(f'Train_MI_Regularization_Loss_{args.attributes[i]}', avg_mi_reg_losses[i], step = epoch)
            comet_experiment.log_metric(f'Train_Demographic_Loss_{args.attributes[i]}', avg_demographic_losses[i], step = epoch)
        comet_experiment.log_metric('Train_Vector_Quantization_Loss', avg_vq_loss, step = epoch)
        comet_experiment.log_metric('Train_Quantizated_CLIP_Loss', avg_quantized_clip_loss, step = epoch)
        comet_experiment.log_metric('Train_Text_Contrastive_Loss', avg_text_contrastive_loss, step = epoch)

        epoch_comet_log_text = ''
        epoch_comet_log_text += f'Train_Total_Loss: {round(avg_loss, 4)}\n'
        for i in range(len(args.attributes)):
            epoch_comet_log_text += f'Train_Adversarial_Loss_{args.attributes[i]}: {round(avg_reverse_adv_losses[i], 4)}\n'
            epoch_comet_log_text += f'Train_MI_Regularization_Loss_{args.attributes[i]}: {round(avg_mi_reg_losses[i], 4)}\n'
            epoch_comet_log_text += f'Train_Demographic_Loss_{args.attributes[i]}: {round(avg_demographic_losses[i], 4)}\n'
        epoch_comet_log_text += f'Train_Vector_Quantization_Loss: {round(avg_vq_loss, 4)}\n'
        epoch_comet_log_text += f'Train_Quantizated_CLIP_Loss: {round(avg_quantized_clip_loss, 4)}\n'
        epoch_comet_log_text += f'Train_Text_Contrastive_Loss: {round(avg_text_contrastive_loss, 4)}\n'

        # iterate over validation dataset
        eval_avg_loss = 0
        all_probs = []
        all_labels = []
        all_attrs = []
        for batch in val_dataloader:
            images,texts, label_and_attributes = batch 

            images= images.to(device)
            texts = texts.to(device)
            glaucoma_labels = label_and_attributes[:, 0].to(device)
            attributes = label_and_attributes[:, 1:].to(device)

            class_text_feats = []
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=1, keepdim=True)

                for i in range(texts.shape[1]):
                    text_features = model.encode_text(texts[:, i, :])
                    text_features /= text_features.norm(dim=1, keepdim=True)
                    class_text_feats.append(text_features[:, None, :])
                # concatentate class_text_feats along the second dimension
                class_text_feats = torch.cat(class_text_feats, dim=1)
                
            vl_prob, vl_logits = compute_vl_prob(image_features, class_text_feats)

            all_probs.append(vl_prob[:,1].cpu().numpy())
            all_labels.append(glaucoma_labels.cpu().numpy())
            all_attrs.append(attributes.cpu().numpy())

            # apply binary cross entropy loss
            loss = F.binary_cross_entropy(vl_prob[:,1].float(), glaucoma_labels.float())
            eval_avg_loss += loss.item()

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_attrs = np.concatenate(all_attrs, axis=0)
        eval_avg_loss /= len(val_dataloader)

        comet_experiment.log_metric('Val_Total_Loss', eval_avg_loss, step = epoch)

        epoch_comet_log_text += f'Val_Total_Loss: {round(eval_avg_loss, 4)}\n'

        logger.log(f'===> epoch[{epoch:03d}/{args.num_epochs:03d}], training loss: {avg_loss:.4f}, eval loss: {eval_avg_loss:.4f}')

        overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(all_probs, all_labels, all_attrs.T)

        if best_auc <= overall_auc:
            best_auc = overall_auc
            best_acc = overall_acc
            best_ep = epoch
            best_auc_groups = eval_aucs_by_attrs
            best_dpd_groups = eval_dpds
            best_eod_groups = eval_eods
            best_es_acc = eval_es_acc
            best_es_auc = eval_es_auc
            best_between_group_disparity = between_group_disparity

            if args.save_all_best_epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_avg_loss,
                    }, os.path.join(args.result_dir, f"clip_ep{epoch:03d}.pth"))
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_avg_loss,
                    }, os.path.join(args.result_dir, f"clip_best.pth"))


        if args.result_dir is not None:
            np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'), 
                        val_pred=all_probs, val_gt=all_labels, val_attr=all_attrs)

        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
        logger.log(best_auc_groups)

        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(avg_loss,4))
        
        logger.logkv('eval_loss', round(eval_avg_loss,4))
        logger.logkv('eval_acc', round(overall_acc,4))

        comet_experiment.log_metric('Accuracy', round(overall_acc * 100, 2), step = epoch)
        epoch_comet_log_text += f'Accuracy: {round(overall_acc * 100, 2)}\n'

        logger.logkv('eval_auc', round(overall_auc,4))

        comet_experiment.log_metric('AUC', round(overall_auc * 100, 2), step = epoch)
        epoch_comet_log_text += f'AUC: {round(overall_auc * 100, 2)}\n'

        for ii in range(len(eval_es_acc)):
            logger.logkv(f'eval_es_acc_attr{ii}', round(eval_es_acc[ii],4))

            comet_experiment.log_metric(f'ES_Accuracy_{idx_to_attr[ii]}', round(eval_es_acc[ii] * 100, 2), step = epoch)
            epoch_comet_log_text += f'ES_Accuracy_{idx_to_attr[ii]}: {round(eval_es_acc[ii] * 100, 2)}\n'

        for ii in range(len(eval_es_auc)):
            logger.logkv(f'eval_es_auc_attr{ii}', round(eval_es_auc[ii],4))

            comet_experiment.log_metric(f'ES_AUC_{idx_to_attr[ii]}', round(eval_es_auc[ii] * 100, 2), step = epoch)
            epoch_comet_log_text += f'ES_AUC_{idx_to_attr[ii]}: {round(eval_es_auc[ii] * 100, 2)}\n'
        
        for ii in range(len(eval_aucs_by_attrs)):
            for iii in range(len(eval_aucs_by_attrs[ii])):
                logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(eval_aucs_by_attrs[ii][iii],4))

                comet_experiment.log_metric(f'AUC_{idx_to_attr[ii]}_{per_attr_idx_to_grp[idx_to_attr[ii]][iii]}', round(eval_aucs_by_attrs[ii][iii] * 100, 2), step = epoch)
                epoch_comet_log_text += f'AUC_{idx_to_attr[ii]}_{per_attr_idx_to_grp[idx_to_attr[ii]][iii]}: {round(eval_aucs_by_attrs[ii][iii] * 100, 2)}\n'

        for ii in range(len(between_group_disparity)):
            logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0],4))
            logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1],4))

        for ii in range(len(eval_dpds)):
            logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii],4))

            comet_experiment.log_metric(f'DPD_{idx_to_attr[ii]}', round(eval_dpds[ii] * 100, 2), step = epoch)
            epoch_comet_log_text += f'DPD_{idx_to_attr[ii]}: {round(eval_dpds[ii] * 100, 2)}\n'

        for ii in range(len(eval_eods)):
            logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii],4))

            comet_experiment.log_metric(f'DEOdds_{idx_to_attr[ii]}', round(eval_eods[ii] * 100, 2), step = epoch)
            epoch_comet_log_text += f'DEOdds_{idx_to_attr[ii]}: {round(eval_eods[ii] * 100, 2)}\n'

        logger.dumpkvs()

        comet_experiment.log_text(epoch_comet_log_text, step = epoch)
    
    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:

                esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
                
                dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')

    os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}')