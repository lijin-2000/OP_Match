import logging
from utils.misc import test, test_ood
import pandas as pd
import os

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0
def eval_model(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, ema_model):
    if args.amp:
        from apex import amp
    global best_acc
    global best_acc_val

    model.eval()
    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model
    epoch = 0
    results = []
    if args.local_rank in [-1, 0]:
        val_acc = test(args, val_loader, test_model, epoch, val=True)
        test_loss, close_valid, test_overall, \
        test_unk, test_roc, test_roc_softm, test_id, test_FPR \
            = test(args, test_loader, test_model, epoch)
        ood_results = {}
        for ood in ood_loaders.keys():
            roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
            ood_results[ood] = roc_ood
            logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))
            logger.info(f"ROC AUC for OOD dataset {ood}: {roc_ood:.3f}")

        overall_valid = test_overall
        unk_valid = test_unk
        roc_valid = test_roc
        roc_softm_valid = test_roc_softm
        epoch_results = {
            "Epoch": epoch,
            "Validation Closed Accuracy": val_acc,
            "Test Closed Accuracy": close_valid,
            "Test Overall Accuracy": overall_valid,
            "Test Unknown Accuracy": unk_valid,
            "Test ROC AUC": roc_valid,
            "Test ROC AUC (Softmax)": roc_softm_valid,
            "Test FPR95 (Softmax)": test_FPR
        }
    
        # 添加每个 OOD 数据集的 ROC AUC 结果
        for ood_name, roc_ood in ood_results.items():
            epoch_results[f"ROC AUC (OOD: {ood_name})"] = roc_ood

        # 将当前 epoch 的结果加入总结果列表
        results.append(epoch_results)

        # 打印结果
        logger.info('validation closed acc: {:.3f}'.format(val_acc))
        logger.info('test closed acc: {:.3f}'.format(close_valid))
        logger.info('test overall acc: {:.3f}'.format(overall_valid))
        logger.info('test unk acc: {:.3f}'.format(unk_valid))
        logger.info('test roc: {:.3f}'.format(roc_valid))
        logger.info('test roc soft: {:.3f}'.format(roc_softm_valid))

        # 获取 args.resume 的上一级目录
        parent_dir = os.path.dirname(args.resume)
        file_path = os.path.join(parent_dir, f"model_evaluation_epoch_{epoch}.csv")

        # 在每个 epoch 结束后将结果保存为 CSV 文件
        df = pd.DataFrame(results)
        df.to_csv(file_path, index=False)
        print(f"模型评估结果已保存到: {file_path}")
    if args.local_rank in [-1, 0]:
        args.writer.close()
