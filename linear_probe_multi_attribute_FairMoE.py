import argparse
import numpy as np
import time
import sys
from pathlib import Path
from typing import Iterable, Optional
import shutil
import datetime

import torch
from torch.utils.data import DataLoader

# from timm.models.layers import trunc_normal_
from timm.layers import trunc_normal_
from timm.data import Mixup

import clip

from src.modules import *
from src.modules import NativeScalerWithGradNormCount as NativeScaler

from FairMoE import *

def get_args_parser():
    parser = argparse.ArgumentParser('Linear probing for image classification using FairMoE', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size during probing')
    parser.add_argument('--epochs', default=90, type=int)

    # Model parameters
    parser.add_argument('--model_arch', default='vit-b16', type=str, help='options: vit-b16 | vit-l14')
    parser.add_argument('--attributes', nargs='+', type=str, default=['race', 'gender', 'ethnicity', 'language'], help='Array includes a combination of race|gender|ethnicity|language')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune_checkpoint', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--dataset_dir', default='./data', type=str)
    parser.add_argument('--nb_classes', default=1, type=int, help='number of the classification output nodes')

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    
    # parser.add_argument("--summary_type", type=str, required=True, default='original', choices=['original', 'pmc-llama', 'med42', 'gpt-3.5-turbo', 'gpt-4'])

    # parser.add_argument("--vl_feats_type", type=str, required=True, choices=['image', 'multimodal'])

    return parser


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, args=None):
    model.train(True)
    # metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)

    optimizer.zero_grad()

    # for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for batch_idx, batch in enumerate(data_loader):
        samples, _, label_and_attributes = batch 
        targets = label_and_attributes[:, 0]
        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # Changed Code
        adjust_learning_rate(optimizer, batch_idx / len(data_loader) + epoch, args)
        #######################

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast(device):
            # if args.vl_feats_type == 'image':
            #    outputs = model.head(model.encode_image(samples))
            image_features, _ = model.encode_image(samples)
            outputs = model.head(image_features)
            # elif args.vl_feats_type == 'multimodal':
            #    clip_text_input = torch.cat([clip.tokenize(truncate_note(tmp_note)) for tmp_note in batch['text_input']]).to(device)
            #    concat_feats = torch.cat([model.module.encode_image(samples), model.module.encode_text(clip_text_input)], dim=1)
            #    outputs = model.module.head(concat_feats)
            loss = torch.nn.BCEWithLogitsLoss()(outputs[:, 0], targets.type(torch.float32))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        # if (data_iter_step + 1) % accum_iter == 0:
        #    optimizer.zero_grad()
        optimizer.zero_grad()

        # torch.cuda.synchronize()

        '''
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
        '''

    '''
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    '''


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    # switch to evaluation mode
    model.eval()

    all_probs = []
    all_labels = []
    all_attrs = []

    for batch in data_loader:
        images, _, label_and_attributes = batch
        target = label_and_attributes[:, 0]
        attributes = label_and_attributes[:, 1:]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.amp.autocast(device):
            # if args.vl_feats_type == 'image':
            #    output = model.module.head(model.module.encode_image(images))
            image_features, _ = model.encode_image(images)
            output = model.head(image_features)
            # elif args.vl_feats_type == 'multimodal':
            #    clip_text_input = torch.cat([clip.tokenize(truncate_note(tmp_note)) for tmp_note in batch['text_input']]).to(device)
            #    concat_feats = torch.cat([model.module.encode_image(images), model.module.encode_text(clip_text_input)], dim=1)
            #    output = model.module.head(concat_feats)
            # loss = torch.nn.BCEWithLogitsLoss()(output, target.type(torch.float32))

        all_probs.append(torch.sigmoid(output[:, 0]).cpu().numpy())
        all_labels.append(target.cpu().numpy())
        all_attrs.append(attributes.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_attrs = np.concatenate(all_attrs, axis=0)

    overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(all_probs, all_labels, all_attrs.T)
    
    test_stats = {
        'overall_acc': overall_acc,
        'eval_es_acc': eval_es_acc,
        'overall_auc': overall_auc,
        'eval_es_auc': eval_es_auc,
        'eval_aucs_by_attrs': eval_aucs_by_attrs,
        'eval_dpds': eval_dpds,
        'eval_eods': eval_eods,
        'between_group_disparity': between_group_disparity
    }

    return test_stats


def main(args):
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # fix the seed for reproducibility
    seed = args.seed
    set_random_seed(seed)

    # the number of groups in each attribute
    groups_in_attrs = [3, 2, 2, 3]
    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}

    model_arch_mapping = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}
    model, preprocess = clip.load(model_arch_mapping[args.model_arch], device = device, jit = False) #Must set jit=False for training
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

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Training', text_source = 'label')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)

    val_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Validation')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)

    test_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)

    print(f'# of training samples: {train_dataset.__len__()}, # of validation samples: {val_dataset.__len__()}, # of testing samples: {test_dataset.__len__()}')

    if args.finetune_checkpoint != '':
        model.load_state_dict(torch.load(args.finetune_checkpoint)["model_state_dict"])
    # if args.vl_feats_type == 'image':
    #    model.head = torch.nn.Linear(embedding_dim, args.nb_classes)
    # elif args.vl_feats_type == 'multimodal':
    #    model.head = torch.nn.Linear(2*embedding_dim, args.nb_classes)
    
    # Changed Code
    # model.head = torch.nn.Linear(embedding_dim, args.nb_classes)
    # trunc_normal_(model.head.weight, std=0.01)
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    model.lin1 = torch.nn.Linear(embedding_dim, 256)
    model.lin2 = torch.nn.Linear(256, 128)
    model.lin3 = torch.nn.Linear(128, 64)
    # model.lin4 = torch.nn.Linear(64, 32)
    # model.lin5 = torch.nn.Linear(32, args.nb_classes)
    model.lin4 = torch.nn.Linear(64, args.nb_classes)
    trunc_normal_(model.lin1.weight, std=0.01)
    trunc_normal_(model.lin2.weight, std=0.01)
    trunc_normal_(model.lin3.weight, std=0.01)
    trunc_normal_(model.lin4.weight, std=0.01)
    # trunc_normal_(model.lin5.weight, std=0.01)
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.lin1.in_features, affine=False, eps=1e-6), 
                                     model.lin1,
                                     torch.nn.BatchNorm1d(model.lin2.in_features, affine=False, eps=1e-6), 
                                     model.lin2,
                                     torch.nn.BatchNorm1d(model.lin3.in_features, affine=False, eps=1e-6), 
                                     model.lin3,
                                     torch.nn.BatchNorm1d(model.lin4.in_features, affine=False, eps=1e-6), 
                                     model.lin4)
    ##########################

    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    # print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number of params: %.2f' % (n_parameters))
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / args.batch_size))
    print("actual lr: %.2e" % args.lr)
    print("batch size: %d" % args.batch_size)

    optimizer = LARS(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    '''
    if args.eval:
        test_stats = evaluate(data_loader_test, model, device)
        print(f"AUC of the network on the {len(dataset_test)} test images: {test_stats['overall_auc']:.1f}%")
        exit(0)
    '''

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_val_auc = -1
    best_epoch_test_stats = None
    for epoch in range(args.epochs):
        train_one_epoch(
            model, train_dataloader,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            args=args
        )

        if (epoch+1)%10 == 0:
            val_stats = evaluate(val_dataloader, model, device, args)
            test_stats = evaluate(test_dataloader, model, device, args)
            print(f"Epoch {epoch+1}:")
            print(f"AUC of the network on the {len(val_dataset)} val images: {val_stats['overall_auc']}")
            print(f"AUC of the network on the {len(test_dataset)} test images: {test_stats['overall_auc']}")
            if val_stats["overall_auc"] > max_val_auc:
                max_val_auc = val_stats["overall_auc"]
                best_epoch_test_stats = test_stats
                if args.output_dir:
                    # misc.save_model_best(
                    #    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    #    loss_scaler=loss_scaler, epoch=epoch)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(args.output_dir, f"clip_best.pth"))
                print('New Best Test Results:')
                print(best_epoch_test_stats)
            print(f"Max Val AUC: {max_val_auc}")
        
        # if (epoch+1)%100 == 0:
        #     shutil.copy(os.path.join(args.output_dir, f"clip_best.pth"), os.path.join(args.output_dir, f"clip_best_{epoch+1}.pth"))

    print("Best Epoch Test Stats:")
    for k, v in best_epoch_test_stats.items():
        print(f"{k}: {v}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)