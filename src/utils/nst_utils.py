import torch
import segmentation_models_pytorch as smp


TEMPERATURE = 10


def generate_pseudo_labels(images_train, images_test, models, confidence_thres):
    """Utility for generating the pseudo-labels using ensembles."""

    # set the models to eval mode
    for model in models:
        model.eval()

    with torch.no_grad():
        # pass train images into models
        preds_1 = models[0](images_train)
        preds_2 = models[1](images_train)
        final_predictions_train = torch.stack((preds_1, preds_2), dim=0).mean(0)

        # pass test images into models
        preds_1 = models[0](images_test)
        preds_2 = models[1](images_test)
        final_predictions_test = torch.stack((preds_1, preds_2), dim=0).mean(0)
        final_predictions_test_, _ = torch.nn.Softmax(dim=1)(
            torch.tensor(final_predictions_test)
        ).max(1)

        # compute thresholding mask
        test_mask = torch.sum(
            final_predictions_test_ > confidence_thres, dim=(1, 2)
        ) > (0.95 * 256 * 256)

        # concatenate all predictions
        all_predictions = torch.cat(
            (final_predictions_train, final_predictions_test), dim=0
        )

    return all_predictions, test_mask


def soft_distillation_loss(predictions, pseudo_labels, ground_truths, alpha=0.1):
    log_softmax = torch.nn.LogSoftmax(dim=1)
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")

    if (alpha > 0.0) & (ground_truths is not None):
        # https://discuss.pytorch.org/t/kldiv-loss-reduction/109131/3
        kl_divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        distillation_loss = kl_divergence(
            log_softmax(predictions / TEMPERATURE),
            log_softmax(pseudo_labels / TEMPERATURE),
        )
        student_loss = criterion_dice(predictions, ground_truths)
        return ((1 - alpha) * distillation_loss) + (alpha * student_loss)
    else:
        kl_divergence = torch.nn.KLDivLoss(reduction="none", log_target=True)
        distillation_loss = kl_divergence(
            log_softmax(predictions / TEMPERATURE),
            log_softmax(pseudo_labels / TEMPERATURE),
        )
        return distillation_loss
