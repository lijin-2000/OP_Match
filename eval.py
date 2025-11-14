import logging
import os
import csv
from utils import test, test_ood

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
    # Use epoch stored in checkpoint if available
    epoch = getattr(args, "start_epoch", 0)
    if args.local_rank in [-1, 0]:
        val_acc = test(args, val_loader, test_model, epoch, val=True)
        test_loss, close_valid, test_overall, \
        test_unk, test_roc, test_roc_softm, test_fpr95_softm, test_id \
            = test(args, test_loader, test_model, epoch)
        ood_results = {}
        for ood in ood_loaders.keys():
            roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
            logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))
            ood_results[ood] = roc_ood

        overall_valid = test_overall
        unk_valid = test_unk
        roc_valid = test_roc
        roc_softm_valid = test_roc_softm
        logger.info('validation closed acc: {:.3f}'.format(val_acc))
        logger.info('test closed acc: {:.3f}'.format(close_valid))
        logger.info('test overall acc: {:.3f}'.format(overall_valid))
        logger.info('test unk acc: {:.3f}'.format(unk_valid))
        logger.info('test roc: {:.3f}'.format(roc_valid))
        logger.info('test roc soft: {:.3f}'.format(roc_softm_valid))
        logger.info('test FPR95 soft: {:.3f}'.format(test_fpr95_softm))

        # Save evaluation results to CSV under the same directory as resume/out
        out_dir = getattr(args, "out", ".")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "eval_results.csv")
        file_exists = os.path.isfile(csv_path)

        # For CIFAR-10, we expect OOD: svhn, cifar100, imagenet
        roc_svhn = ood_results.get("svhn", 0.0)
        roc_cifar100 = ood_results.get("cifar100", 0.0)
        roc_imagenet = ood_results.get("imagenet", 0.0)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Epoch",
                    "Validation Closed Accuracy",
                    "Test Closed Accuracy",
                    "Test Overall Accuracy",
                    "Test Unknown Accuracy",
                    "Test ROC AUC",
                    "Test ROC AUC (Softmax)",
                    "Test FPR95 (Softmax)",
                    "ROC AUC (OOD: svhn)",
                    "ROC AUC (OOD: cifar100)",
                    "ROC AUC (OOD: imagenet)",
                ])
            writer.writerow([
                int(epoch),
                float(val_acc),
                float(close_valid),
                float(overall_valid),
                float(unk_valid),
                float(roc_valid),
                float(roc_softm_valid),
                float(test_fpr95_softm),
                float(roc_svhn),
                float(roc_cifar100),
                float(roc_imagenet),
            ])
    if args.local_rank in [-1, 0]:
        args.writer.close()
