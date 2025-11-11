import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from tqdm import tqdm
from utils import AverageMeter, ova_loss,\
    save_checkpoint, ova_ent, \
    test, test_ood, exclude_dataset
import os
import csv
import gc
import torch
from utils.misc import get_block3_params

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0

def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp

    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"

    model.train()
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformOpenMatch
    elif args.dataset == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformOpenMatch
    elif 'imagenet' in args.dataset:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_Weak


    unlabeled_dataset_all.transform = func_trans(mean=mean, std=std)
    labeled_dataset = copy.deepcopy(labeled_trainloader.dataset)
    labeled_dataset.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler


    for epoch in range(args.start_epoch, args.epochs):
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        
    
        block3_params = get_block3_params(model)
        # 初始化每个损失的梯度累积器
        epoch_grads = {}
        loss_names = ['x', 'o', 'oem', 'oem_id', 'oem_ood', 'socr', 'socr_id', 'socr_ood', 'fix', 'fix_id', 'fix_ood']
        for name in loss_names:
            epoch_grads[name] = [torch.zeros_like(p).to(args.device) for p in block3_params]
        
        batch_count = 0
        
        # 定义一个函数来计算损失的梯度而不影响当前计算图
        def compute_individual_grad(loss, params):
            # 使用torch.autograd.grad计算梯度而不修改计算图
            if not hasattr(loss, 'grad_fn') or loss.grad_fn is None:
                # 如果损失没有连接到计算图，直接返回零梯度
                return [torch.zeros_like(p) for p in params]
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            # 处理可能为None的梯度
            return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

        if epoch >= args.start_fix:
            ## pick pseudo-inliers
            exclude_dataset(args, unlabeled_dataset, ema_model.ema)


        unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                           sampler = train_sampler(unlabeled_dataset),
                                           batch_size = args.batch_size * args.mu,
                                           num_workers = args.num_workers,
                                           drop_last = True)
        unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                           sampler=train_sampler(unlabeled_dataset_all),
                                           batch_size=args.batch_size * args.mu,
                                           num_workers=args.num_workers,
                                           drop_last=True)

        unlabeled_iter = iter(unlabeled_trainloader)
        unlabeled_all_iter = iter(unlabeled_trainloader_all)

        for batch_idx in range(args.eval_step):
            ## Data loading
            try:
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s, _), targets_u_fix = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), targets_u_fix = unlabeled_iter.next()
            try:
                (inputs_all_w, inputs_all_s, _), targets_all = unlabeled_all_iter.next()
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, _), targets_all = unlabeled_all_iter.next()
            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]

            inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
            inputs = torch.cat([inputs_x, inputs_x_s,
                                inputs_all], 0).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_all = targets_all.to(args.device)
            targets_u_fix = targets_u_fix.to(args.device)
            ## Feed data with decoupled projections
            logits_id_all, logits_open_all, z_id_all, z_ood_all = model(inputs, feature=True)
            logits_open_u1, logits_open_u2 = logits_open_all[2*b_size:].chunk(2)

            ## Loss for labeled samples (classification uses z_id head)
            Lx = F.cross_entropy(logits_id_all[:2*b_size],
                                      targets_x.repeat(2), reduction='mean')
            # Open-set OVA loss on decoupled open head (z_ood)
            Lo = ova_loss(logits_open_all[:2*b_size], targets_x.repeat(2))

            ## Open-set entropy minimization with soft responsibilities
            # prepare open-head probabilities for unlabeled weak/strong views
            logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            open_u1 = F.softmax(logits_open_u1, 1)
            open_u2 = F.softmax(logits_open_u2, 1)

            # predicted closed class from z_id classifier
            logits_all_id = logits_id_all[2*b_size:]
            logits_all_u1, logits_all_u2 = logits_all_id.chunk(2)
            pred_u1 = torch.argmax(logits_all_u1.detach(), dim=1)
            pred_u2 = torch.argmax(logits_all_u2.detach(), dim=1)
            idx_u1 = torch.arange(open_u1.size(0), device=inputs.device)
            idx_u2 = torch.arange(open_u2.size(0), device=inputs.device)
            unk_u1 = open_u1[idx_u1, 0, pred_u1]
            unk_u2 = open_u2[idx_u2, 0, pred_u2]
            p_id_u1 = 1.0 - unk_u1
            p_id_u2 = 1.0 - unk_u2
            p_ood_u1 = 1.0 - p_id_u1
            p_ood_u2 = 1.0 - p_id_u2

            # per-sample open-head entropy
            ent_u1 = torch.sum(torch.sum(-open_u1 * torch.log(open_u1 + 1e-8), dim=1), dim=1)
            ent_u2 = torch.sum(torch.sum(-open_u2 * torch.log(open_u2 + 1e-8), dim=1), dim=1)
            L_oem_id = 0.5 * (torch.mean(p_id_u1 * ent_u1) + torch.mean(p_id_u2 * ent_u2))
            L_oem_ood = 0.5 * (torch.mean(p_ood_u1 * ent_u1) + torch.mean(p_ood_u2 * ent_u2))
            L_oem = L_oem_id - L_oem_ood

            ## Soft consistency regularization with responsibilities
            L_socr_per = torch.sum(torch.sum(torch.abs(open_u1 - open_u2)**2, 1), 1)
            L_socr_id = 0.5 * (torch.mean(p_id_u1 * L_socr_per) + torch.mean(p_id_u2 * L_socr_per))
            L_socr_ood = 0.5 * (torch.mean(p_ood_u1 * L_socr_per) + torch.mean(p_ood_u2 * L_socr_per))
            L_socr = L_socr_id + L_socr_ood

            if epoch >= args.start_fix:
                inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
                logits_id_fix_all, logits_open_fix, z_id_fix, z_ood_fix = model(inputs_ws, feature=True)
                logits_u_w, logits_u_s = logits_id_fix_all.chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                L_fix = (F.cross_entropy(logits_u_s,
                                         targets_u,
                                         reduction='none') * mask).mean()
                mask_probs.update(mask.mean().item())

            else:
                L_fix = torch.zeros(1).to(args.device).mean()
                
                
            # 使用软责任分项（不依赖GT），已在上方计算 L_oem_id/L_oem_ood 与 L_socr_id/L_socr_ood
            
            # FixMatch分项的统计留空（已换为软路由方案，必要时可复用 p_id/p_ood）
            

            batch_count += 1
            # 首先确保所有参数的梯度为零，因为我们要进行梯度统计
            optimizer.zero_grad() 
            # 计算Lx的梯度
            if 'Lx' in locals():
                lx_grads = compute_individual_grad(Lx, block3_params)
                for i in range(len(epoch_grads['x'])):
                    epoch_grads['x'][i] += lx_grads[i]
            
            # 计算Lo的梯度
            if 'Lo' in locals():
                lo_grads = compute_individual_grad(Lo, block3_params)
                for i in range(len(epoch_grads['o'])):
                    epoch_grads['o'][i] += lo_grads[i]
            
            # 计算L_oem的梯度（已按软责任分裂并合成）
            if 'L_oem' in locals():
                oem_grads = compute_individual_grad(args.lambda_oem * L_oem, block3_params)
                for i in range(len(epoch_grads['oem'])):
                    epoch_grads['oem'][i] += oem_grads[i]
            
            # 计算L_oem_id和L_oem_ood的梯度
            if 'L_oem_id' in locals():
                oem_id_grads = compute_individual_grad(args.lambda_oem * L_oem_id, block3_params)
                for i in range(len(epoch_grads['oem_id'])):
                    epoch_grads['oem_id'][i] += oem_id_grads[i]
            if 'L_oem_ood' in locals():
                oem_ood_grads = compute_individual_grad(args.lambda_oem * L_oem_ood, block3_params)
                for i in range(len(epoch_grads['oem_ood'])):
                    epoch_grads['oem_ood'][i] += oem_ood_grads[i]
            
            # 计算L_socr的梯度
            if 'L_socr' in locals():
                socr_grads = compute_individual_grad(args.lambda_socr * L_socr, block3_params)
                for i in range(len(epoch_grads['socr'])):
                    epoch_grads['socr'][i] += socr_grads[i]
            
            # 计算L_socr_id和L_socr_ood的梯度
            if 'L_socr_id' in locals():
                socr_id_grads = compute_individual_grad(args.lambda_socr * L_socr_id, block3_params)
                for i in range(len(epoch_grads['socr_id'])):
                    epoch_grads['socr_id'][i] += socr_id_grads[i]
            if 'L_socr_ood' in locals():
                socr_ood_grads = compute_individual_grad(args.lambda_socr * L_socr_ood, block3_params)
                for i in range(len(epoch_grads['socr_ood'])):
                    epoch_grads['socr_ood'][i] += socr_ood_grads[i]
            
            # 计算L_fix的梯度
            if 'L_fix' in locals() and epoch >= args.start_fix:
                fix_grads = compute_individual_grad(L_fix, block3_params)
                for i in range(len(epoch_grads['fix'])):
                    epoch_grads['fix'][i] += fix_grads[i]
            
            # 计算L_fix_id和L_fix_ood的梯度
            if 'L_fix_id' in locals() and epoch >= args.start_fix:
                fix_id_grads = compute_individual_grad(L_fix_id, block3_params)
                for i in range(len(epoch_grads['fix_id'])):
                    epoch_grads['fix_id'][i] += fix_id_grads[i]
            if 'L_fix_ood' in locals() and epoch >= args.start_fix:
                fix_ood_grads = compute_individual_grad(L_fix_ood, block3_params)
                for i in range(len(epoch_grads['fix_ood'])):
                    epoch_grads['fix_ood'][i] += fix_ood_grads[i]
            
            # Decoupling regularizer
            lambda_dec = getattr(args, 'lambda_dec', 0.0)
            if lambda_dec > 0:
                zid = F.normalize(z_id_all, dim=1)
                zood = F.normalize(z_ood_all, dim=1)
                dot_sq = (zid * zood).sum(dim=1).pow(2)
                L_dec = dot_sq.mean()
            else:
                L_dec = torch.zeros(1).to(args.device).mean()

            loss = Lx + Lo + args.lambda_oem * L_oem  \
                   + args.lambda_socr * L_socr + L_fix \
                   + lambda_dec * L_dec
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_oem"] = losses_oem.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]


            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()
        
        # 在epoch结束后，计算参数更新变化与各损失函数梯度的余弦相似度，以及不同损失之间梯度的余弦相似度
        if batch_count > 0 :
            # print(f"计算epoch {epoch} 不同损失之间梯度的余弦相似度...")
            try:
                # 计算每个损失的平均梯度（除以batch数量）
                avg_grads = {}
                for name, grads in epoch_grads.items():
                    # 只有当这个损失在当前epoch中被计算过才进行平均
                    if any(g.abs().sum().item() > 0 for g in grads):
                        avg_grad = [g / batch_count for g in grads]
                        avg_grads[name] = avg_grad
                # 创建一个字典保存所有相似度结果
                all_similarities = {}
                all_similarities['epoch'] = epoch
                
                # 计算不同损失之间梯度的余弦相似度
                loss_names = list(avg_grads.keys())
                for i in range(len(loss_names)):
                    for j in range(i+1, len(loss_names)):
                        loss1_name = loss_names[i]
                        loss2_name = loss_names[j]
                        # 计算两个损失梯度之间的余弦相似度
                        sim = cosine_similarity(avg_grads[loss1_name], avg_grads[loss2_name])
                        all_similarities[f'sim_{loss1_name}_{loss2_name}'] = sim
                        # print(f"Epoch {epoch}: {loss1_name}梯度与{loss2_name}梯度的余弦相似度 = {sim}")
                
                # 保存相似度到文件
                save_epoch_update_similarities(args.out, all_similarities)
                
            except Exception as e:
                print(f"计算相似度时出错: {e}")
            

            torch.cuda.empty_cache()

        if not args.no_progress:
            p_bar.close()
            
        # 这里不再需要重复计算，因为前面已经使用累积的平均梯度计算了相似度

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            val_acc = test(args, val_loader, test_model, epoch, val=True)
            test_loss, test_acc_close, test_overall, \
            test_unk, test_roc, test_roc_softm, test_id \
                = test(args, test_loader, test_model, epoch)

            for ood in ood_loaders.keys():
                roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
                logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_o', losses_o.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_oem', losses_oem.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_socr', losses_socr.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_fix', losses_fix.avg, epoch)
            args.writer.add_scalar('train/6.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = val_acc > best_acc_val
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                overall_valid = test_overall
                close_valid = test_acc_close
                unk_valid = test_unk
                roc_valid = test_roc
                roc_softm_valid = test_roc_softm
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'acc overall': test_overall,
                'unk': test_unk,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            logger.info('Valid overall acc: {:.3f}'.format(overall_valid))
            logger.info('Valid unk acc: {:.3f}'.format(unk_valid))
            logger.info('Valid roc: {:.3f}'.format(roc_valid))
            logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
    if args.local_rank in [-1, 0]:
        args.writer.close()


# 添加在文件顶部的导入语句之后
def cosine_similarity(grad1, grad2):
    # 将列表中的多个梯度张量拼接成一个一维张量
    grad1_tensor = torch.cat([g.reshape(-1) for g in grad1]) if isinstance(grad1, list) else grad1
    grad2_tensor = torch.cat([g.reshape(-1) for g in grad2]) if isinstance(grad2, list) else grad2

    # 计算余弦相似度
    dot_product = torch.dot(grad1_tensor, grad2_tensor)
    norm1 = torch.norm(grad1_tensor)
    norm2 = torch.norm(grad2_tensor)
    return (dot_product / (norm1 * norm2 + 1e-8)).item()

def vectorize_grads(grads_list):
    # 如果梯度是列表，拼接成一维向量；否则直接返回
    if isinstance(grads_list, list):
        return torch.cat([g.reshape(-1) for g in grads_list])
    return grads_list

def save_epoch_update_similarities(save_dir, similarities):
    """保存每个epoch参数更新与各损失函数梯度的余弦相似度"""
    # 创建保存目录
    sim_dir = os.path.join(save_dir, "epoch_update_similarities")
    os.makedirs(sim_dir, exist_ok=True)
    
    # 定义保存文件路径
    sim_file = os.path.join(sim_dir, "update_similarities.csv")
    
    # 检查文件是否存在，不存在则创建并写入表头
    file_exists = os.path.isfile(sim_file)
    
    with open(sim_file, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=similarities.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(similarities)
    
    return similarities

# 在损失计算完成后，反向传播之前
