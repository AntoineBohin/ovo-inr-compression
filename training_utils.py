import copy
import os
import time
import warnings
from collections.abc import Mapping

import numpy as np
import skimage.metrics
import torch
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from datasets import model_l1


############################### TRAINING LOOP FUNCTION ##################################

def train(model: torch.nn.Module, train_dataloader: DataLoader, epochs: int, lr: float, steps_til_summary: int, 
          epochs_til_checkpoint: int, model_dir: str, loss_fn: callable, summary_fn: callable, 
          loss_schedules: dict = None, weight_decay: float = 0, l1_reg: float = 0, 
          l1_loss_fn: callable = model_l1, use_amp: bool = True, patience: int = 500,
          warmup: int = 0, early_stop_epochs: int = 5000):
    """
    Training loop for a model with optional L1 regularization and learning rate warmup. Based on https://github.com/vsitzmann/siren
    """
    optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)

    if warmup > 0:
        scheduler = ReduceLROnPlateauWithWarmup(optim, warmup_end_lr=lr, warmup_steps=warmup, mode='min', factor=0.5,
                                                patience=patience, threshold=0.0001, threshold_mode='rel', 
                                                cooldown=0, eps=1e-08, verbose=True)
    else:
        scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=patience, threshold=0.0001,
                                      threshold_mode='rel', cooldown=0, eps=1e-08, verbose=True)
    print("Training model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    cond_mkdir(summaries_dir)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    cond_mkdir(checkpoints_dir)
    writer = SummaryWriter(summaries_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_steps = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        epochs_since_improvement = 0
        best_total_epoch_loss = float("Inf")
        for epoch in range(epochs):
            total_epoch_loss = 0
            if epochs_since_improvement > early_stop_epochs:
                break  # stop early if no improvement since early_stop_epochs epochs

            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    start_time = time.time()

                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)
                    if l1_reg > 0:
                        l1_loss = l1_loss_fn(model, l1_reg)
                        losses = {**losses, **l1_loss}

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps),
                                              total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f, best epoch loss %0.6f" % (
                            epoch, total_epoch_loss, time.time() - start_time, best_total_epoch_loss))
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current_.pth'))

                        summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    optim.zero_grad()
                    scaler.scale(train_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    pbar.update(1)
                    scheduler.step(train_loss)

                total_steps += 1
                total_epoch_loss += train_loss.item()

            epochs_since_improvement += 1
            if total_epoch_loss < best_total_epoch_loss:
                best_total_epoch_loss = total_epoch_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        torch.save(best_state_dict,
                   os.path.join(checkpoints_dir, 'model_best_.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        model.load_state_dict(best_state_dict, strict=True)


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    """
    Custom learning rate scheduler with warmup phase.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, mode: str = 'min', factor: float = 0.1, patience: int = 10, 
                 threshold: float = 1e-4, threshold_mode: str = 'rel', cooldown: int = 0, min_lr: float = 0, 
                 eps: float = 1e-8, warmup_end_lr: float = 0, warmup_steps: int = 0, verbose: bool = False):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, threshold=threshold,
                         threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)
        self.warmup_end_lr = warmup_end_lr
        self.warmup_steps = warmup_steps
        self._set_warmup_lr(1)

    def _set_warmup_lr(self, epoch: int):
        """Set learning rate during warmup phase."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = epoch * (self.warmup_end_lr / self.warmup_steps)
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: increase learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def step(self, metrics: float, epoch: int = None):
        """Override step method to include warmup."""
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.warmup_steps > 0 and epoch <= self.warmup_steps:
            self._set_warmup_lr(epoch)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]



################################## LOSSES #####################################

def image_mse(model_output: dict, ground_truth: dict) -> dict:
    """Computes the Mean Squared Error between the model output and ground_truth."""
    return {'img_loss': ((model_output['model_out'] - ground_truth['img']) ** 2).mean()}
    
def l2_loss(prediction: dict, ground_truth: torch.Tensor) -> torch.Tensor:
    """Computes the L2 loss between the prediction and ground truth."""
    return ((prediction['model_out'] - ground_truth) ** 2).mean()

def model_l1_diff(ref_model: torch.nn.Module, model: torch.nn.Module, l1_lambda: float) -> dict:
    """Computes the L1 norm of the parameter difference between a model and a ref_model and weights it with l1_lambda."""
    l1_norm = sum((p - ref_p).abs().sum() for (p, ref_p) in zip(model.parameters(), ref_model.parameters()))
    return {'l1_loss': l1_lambda * l1_norm}


################################## UTILS EVAL #####################################

def lin2img(tensor: torch.Tensor, image_resolution: tuple = None) -> torch.Tensor:
    """Convert a linear tensor to an image tensor with specified resolution."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor)
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]
    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def lin2img_single(tensor: torch.Tensor, image_resolution: tuple = None) -> torch.Tensor:
    """Convert a single linear tensor to an image tensor."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor)
    return tensor.permute(2, 0, 1)

def check_metrics_full(test_loader: DataLoader, model: torch.nn.Module, image_resolution: tuple) -> tuple:
    """Evaluate model performance on a test dataset and compute metrics."""
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            predictions = model(model_input)
            gt_img = lin2img(gt['img'], image_resolution)
            pred_img = lin2img(predictions['model_out'], image_resolution)
            pred_img = pred_img.detach().cpu().numpy()[0]
            gt_img = gt_img.detach().cpu().numpy()[0]
            p = pred_img.transpose(1, 2, 0)
            trgt = gt_img.transpose(1, 2, 0)
            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5
            mse = skimage.metrics.mean_squared_error(p, trgt)
            ssim = skimage.metrics.structural_similarity(p, trgt, channel_axis=-1, data_range=1)
            psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
    return mse, ssim, psnr

def compute_metrics(model_out: torch.Tensor, gt: torch.Tensor, image_resolution: tuple) -> tuple:
    """Compute evaluation metrics for model output and ground truth."""
    with torch.no_grad():
        gt_img = lin2img(gt, image_resolution)
        pred_img = lin2img(model_out, image_resolution)
        pred_img = pred_img.detach().cpu().numpy()[0]
        gt_img = gt_img.detach().cpu().numpy()[0]
        p = pred_img.transpose(1, 2, 0)
        trgt = gt_img.transpose(1, 2, 0)
        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5
        mse = skimage.metrics.mean_squared_error(p, trgt)
        ssim = skimage.metrics.structural_similarity(p, trgt, channel_axis=-1, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
    return mse, ssim, psnr

def compute_metrics_single(model_out: torch.Tensor, gt: torch.Tensor, image_resolution: tuple) -> tuple:
    """Compute evaluation metrics for a single model output and ground truth."""
    with torch.no_grad():
        # Convert outputs to image tensors
        gt_img = lin2img_single(gt, image_resolution).cpu().numpy()
        pred_img = lin2img_single(model_out, image_resolution).cpu().numpy()

        # Convert from (C, H, W) to (H, W, C)
        p = pred_img.transpose(1, 2, 0)
        trgt = gt_img.transpose(1, 2, 0)

        # Normalize to range [0, 1]
        p = np.clip((p / 2.0) + 0.5, 0.0, 1.0)
        trgt = np.clip((trgt / 2.0) + 0.5, 0.0, 1.0)

        # Calculate metrics
        mse = skimage.metrics.mean_squared_error(p, trgt)
        ssim = skimage.metrics.structural_similarity(p, trgt, channel_axis=-1, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

    return mse, psnr, ssim


############################ UTILS FILE MANAGEMENT ##############################


def cond_mkdir(path: str):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def dict_to_gpu(ob):
    """Recursively move a dictionary of tensors to GPU."""
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    return ob.cuda()

def get_base_overfitting_experiment_folder(CONFIG) -> str:
    """Create string with experiment name and get number of experiment."""
    exp_name = '_'.join([
        CONFIG.dataset, 'epochs' + str(CONFIG.epochs), 'hidden_dims' + str(CONFIG.hidden_dims)])

    exp_folder = os.path.join(CONFIG.exp_root, exp_name)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    return exp_folder

def get_maml_overfitting_experiment_folder(CONFIG, subfolder: str = None) -> str:
    """Create string with experiment name and get number of experiment."""
    exp_name = '_'.join([
        CONFIG.dataset, 'MAML', CONFIG.maml_dataset, 'epochs' + str(CONFIG.epochs), 'hidden_dims' + str(CONFIG.hidden_dims)])
    if subfolder:
        exp_folder = os.path.join(CONFIG.exp_root, subfolder, exp_name)
    else:
        exp_folder = os.path.join(CONFIG.exp_root, exp_name)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    return exp_folder

def get_maml_folder(CONFIG) -> str:
    """Create string with experiment name and get number of experiment."""
    exp_name = '_'.join([
        'MAML', CONFIG.maml_dataset, 'hidden_dim' + str(CONFIG.hidden_dims), 'epochs' + str(CONFIG.maml_epochs) ])

    exp_folder = os.path.join(CONFIG.exp_root, 'maml', exp_name)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    return exp_folder

def rescale_img(x: torch.Tensor, mode: str = 'scale', perc: float = None, tmax: float = 1.0, tmin: float = 0.0) -> torch.Tensor:
    """Rescale image tensor based on the specified mode."""
    if mode == 'scale':
        if perc is not None:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        else:
            xmin, xmax = torch.min(x), torch.max(x)
        
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif mode == 'clamp':
        x = torch.clamp(x, 0, 1)
    return x

def write_psnr(pred_img: torch.Tensor, gt_img: torch.Tensor, writer: SummaryWriter, iter: int, prefix: str):
    """Write PSNR and SSIM metrics to TensorBoard."""
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5
        ssim = skimage.metrics.structural_similarity(p, trgt, channel_axis=-1, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)

def write_image_summary(image_resolution: tuple, model: torch.nn.Module, model_input: dict, gt: dict,
                        model_output: dict, writer: SummaryWriter, total_steps: int, prefix: str = 'train_'):
    """Write image summary to TensorBoard."""
    gt_img = lin2img(gt['img'], image_resolution)
    pred_img = lin2img(model_output['model_out'], image_resolution)
    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    pred_img = rescale_img((pred_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(
        0).detach().cpu().numpy()

    gt_img = rescale_img((gt_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    writer.add_image(prefix + 'pred_img', torch.from_numpy(pred_img).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_img', torch.from_numpy(gt_img).permute(2, 0, 1), global_step=total_steps)

    write_psnr(lin2img(model_output['model_out'], image_resolution),
               lin2img(gt['img'], image_resolution), writer, total_steps, prefix + 'img_')

def plot_sample_image(img_batch: torch.Tensor, ax, image_resolution: tuple):
    """Plot a sample image from a batch."""
    img = lin2img(img_batch, image_resolution)[0].detach().cpu().numpy()
    img += 1
    img /= 2.
    img = np.clip(img, 0., 1.)
    ax.set_axis_off()
    ax.imshow(np.transpose(img, (1, 2, 0)))