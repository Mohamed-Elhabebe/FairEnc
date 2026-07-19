import os
import numpy as np
import argparse
import json

import clip

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
sys.path.append('.')

from src.modules import *
from src import logger

from FairMoE import *

parser = argparse.ArgumentParser(description='FairMoE Linear Probing Testing')

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--eval_set', default='test', type=str, help='options: val | test')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='vit-b16', type=str, help='options: vit-b16 | vit-l14')
parser.add_argument('--attributes', nargs='+', type=str, default=['race', 'gender', 'ethnicity', 'language'], help='Array includes a combination of race|gender|ethnicity|language')
parser.add_argument('--nb_classes', default=1, type=int, help='number of the classification output nodes')
parser.add_argument('--pretrained_weights', default='', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    logger.configure(dir=args.result_dir, log_suffix='eval')

    with open(os.path.join(args.result_dir, f'args_eval.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # the number of groups in each attribute
    groups_in_attrs = [3, 2, 2, 3]
    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}

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
    model.to('cpu')

    n_experts = 0
    for attribute in args.attributes:
        n_experts = n_experts + groups_in_attrs[attr_to_idx[attribute]]

    # moe
    model = FairMoE(model, args.model_arch, n_experts)
    model.to(device)
    if args.model_arch == 'vit-b16':
        embedding_dim = 512
    elif args.model_arch == 'vit-l14':
        embedding_dim = 768
    # model.head = torch.nn.Linear(embedding_dim, args.nb_classes)
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    model.lin1 = torch.nn.Linear(embedding_dim, 256)
    model.lin2 = torch.nn.Linear(256, 128)
    model.lin3 = torch.nn.Linear(128, 64)
    # model.lin4 = torch.nn.Linear(64, 32)
    model.lin4 = torch.nn.Linear(64, args.nb_classes)
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.lin1.in_features, affine=False, eps=1e-6), 
                                     model.lin1,
                                     torch.nn.BatchNorm1d(model.lin2.in_features, affine=False, eps=1e-6), 
                                     model.lin2,
                                     torch.nn.BatchNorm1d(model.lin3.in_features, affine=False, eps=1e-6), 
                                     model.lin3,
                                     torch.nn.BatchNorm1d(model.lin4.in_features, affine=False, eps=1e-6), 
                                     model.lin4)
    model.to(device)

    val_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Validation')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    logger.log(f'# of testing samples: {test_dataset.__len__()}')

    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(test_dataset)
    logger.log(f'group size on race in test set: {group_size_on_race}')
    logger.log(f'group size on gender in test set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in test set: {group_size_on_ethnicity}')

    '''
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 


    if device == "cpu":
      model.float()
    else :
      clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16
    '''

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
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

    for epoch in range(1):
        avg_loss = 0
        
        # iterate over test dataset
        eval_avg_loss = 0
        all_probs = []
        all_labels = []
        all_attrs = []
        for batch in test_dataloader:
            images, _, label_and_attributes = batch 

            images= images.to(device)
            glaucoma_labels = label_and_attributes[:, 0].to(device)
            attributes = label_and_attributes[:, 1:].to(device)

            with torch.amp.autocast(device):
                image_features, _ = model.encode_image(images)
                output = model.head(image_features)

            all_probs.append(torch.sigmoid(output[:, 0]).detach().cpu().numpy())
            all_labels.append(glaucoma_labels.cpu().numpy())
            all_attrs.append(attributes.cpu().numpy())

            # apply binary cross entropy loss
            loss = F.binary_cross_entropy(torch.sigmoid(output[:, 0]).float(), glaucoma_labels.float())
            eval_avg_loss += loss.item()

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_attrs = np.concatenate(all_attrs, axis=0)
        eval_avg_loss /= len(test_dataloader)

        logger.log(f'===> eval loss: {eval_avg_loss:.4f}')

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
        
            '''
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_avg_loss,
                }, os.path.join(args.result_dir, f"clip_ep{epoch:03d}.pth"))
            '''


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
        logger.logkv('eval_auc', round(overall_auc,4))

        for ii in range(len(eval_es_acc)):
            logger.logkv(f'eval_es_acc_attr{ii}', round(eval_es_acc[ii],4))
        for ii in range(len(eval_es_auc)):
            logger.logkv(f'eval_es_auc_attr{ii}', round(eval_es_auc[ii],4))
        for ii in range(len(eval_aucs_by_attrs)):
            for iii in range(len(eval_aucs_by_attrs[ii])):
                logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(eval_aucs_by_attrs[ii][iii],4))

        for ii in range(len(between_group_disparity)):
            logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0],4))
            logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1],4))

        for ii in range(len(eval_dpds)):
            logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii],4))
        for ii in range(len(eval_eods)):
            logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii],4))

        logger.dumpkvs()
    
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

                path_str = f'{args.result_dir}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')

    os.rename(args.result_dir, f'{args.result_dir}_auc{best_auc:.4f}')
