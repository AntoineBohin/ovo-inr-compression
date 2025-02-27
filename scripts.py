import os
import csv
import shutil
import copy
import glob
import json
from collections import OrderedDict
from functools import partial
import re

from torch import nn
import statistics

import torch
import yaml
from torch.utils.data import DataLoader
from types import SimpleNamespace


#import modules
import training_utils

import matplotlib.colors as colors
import numpy as np
import scipy.ndimage
import scipy.special
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import matplotlib.pyplot as plt

from models import INRNet, MAML
from datasets import CelebA, DIV2K, Implicit2DWrapper, ImageFile, model_l1
from training_utils import l2_loss, image_mse, model_l1_diff, check_metrics_full, get_base_overfitting_experiment_folder, get_maml_overfitting_experiment_folder, get_maml_folder, compute_metrics, write_image_summary, dict_to_gpu, plot_sample_image, compute_metrics_single


################################################################################
######################### BASIC OVERFITTING EXPERIMENT #########################
################################################################################

def basic_overfitting_experiment(CONFIG):
    config_dict = vars(CONFIG)
    # ====== Load Dataset ======
    # Get all image file paths in the specified dataset folder
    image_paths = glob.glob(os.path.join(CONFIG.data_root, CONFIG.dataset, '*'))

    # Initialize dictionaries to store performance metrics
    mses, psnrs, ssims = {}, {}, {}

    # ====== Create Experiment Folder ======
    # Determine the folder where experiment results will be saved
    experiment_folder = get_base_overfitting_experiment_folder(CONFIG)
    # Save the script arguments (CONFIG) as a YAML file for reproducibility
    yaml.dump(config_dict, open(os.path.join(experiment_folder, 'CONFIG.yml'), 'w'))

    # ----------------------------------------------------------------------------------------------------

    # ====== Process Each Image : Create one model for each image and store the weights ======
    for idx, image_path in enumerate(image_paths):
        print(f'Processing Image {idx + 1}/{len(image_paths)}: {image_path}')

        # Define the folder where model checkpoints and results will be saved
        image_name = os.path.basename(image_path).split('.')[0]
        root_path = os.path.join(experiment_folder, image_name)
        if os.path.exists(os.path.join(root_path, 'checkpoints', 'model_final.pth')):
            print(f"Skipping {image_name}, model already trained.")
            continue

        # ====== Create DataLoader ======
        # Convert image into a dataset of coordinate-value pairs
        img_dataset = ImageFile(image_path)
        # Open the image using PIL to get its dimensions
        img = PIL.Image.open(image_path)
        image_resolution = (img.size[1] // CONFIG.downscaling_factor, img.size[0] // CONFIG.downscaling_factor)
        # Convert it into a coordinate dataset
        coord_dataset = Implicit2DWrapper(img_dataset, sidelength=image_resolution)

        # DataLoader for batching and shuffling the data
        dataloader = DataLoader(coord_dataset, shuffle=True, pin_memory=True, num_workers=0)


        # ====== Define the Model ======
        # Initialize an INRNet with the specified architecture and encoding
        model = INRNet(
            type=CONFIG.activation,
            mode=CONFIG.encoding,
            sidelength=image_resolution,
            out_features=img_dataset.img_channels,
            hidden_features=CONFIG.hidden_dims,
            num_hidden_layers=CONFIG.hidden_layers,
            encoding_scale=CONFIG.encoding_scale,
            ff_dims=CONFIG.ff_dims
        ).to(device)

        # Print the number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters: {num_params}")

        # ====== Define the Loss Function ======
        # Use MSE (Mean Squared Error) as the primary loss function
        loss_fn = image_mse 
        # Define additional loss functions
        summary_fn = partial(write_image_summary, image_resolution)  # Summary function for logging
        l1_loss_fn = model_l1  # L1 regularization function

        # ====== Train the Model ======
        # Call the main training loop with all required arguments
        training_utils.train(
            model=model,
            train_dataloader=dataloader,
            epochs=CONFIG.epochs,
            lr=CONFIG.lr,
            steps_til_summary=CONFIG.steps_til_summary,
            epochs_til_checkpoint=CONFIG.epochs_til_ckpt,
            model_dir=root_path,
            loss_fn=loss_fn,
            summary_fn=summary_fn,
            l1_reg=CONFIG.l1_reg,
            l1_loss_fn=l1_loss_fn,
            patience=CONFIG.patience
        )

        # ====== Evaluate the Model ======
        # Switch to evaluation mode (disables dropout, batch norm, etc.)
        model.eval()

        # Compute performance metrics: MSE, SSIM, and PSNR
        mse, ssim, psnr = check_metrics_full(dataloader, model, image_resolution)

        # Store metrics in dictionaries
        mses[image_name] = mse
        psnrs[image_name] = psnr
        ssims[image_name] = ssim

    # ====== Compute Average Metrics ======
    # Calculate the average of each metric across all processed images
    metrics = {
        'mse': mses,
        'psnr': psnrs,
        'ssim': ssims,
        'avg_mse': statistics.mean(mses.values()) if mses else 0.0,
        'avg_psnr': statistics.mean(psnrs.values()) if psnrs else 0.0,
        'avg_ssim': statistics.mean(ssims.values()) if ssims else 0.0
    }

    # Save metrics as a JSON file
    with open(os.path.join(experiment_folder, 'result.json'), 'w') as file:
        json.dump(metrics, file, default=str)

    print("\nTraining complete. Results saved to:", experiment_folder)




#################################################################
######################### MAML TRAINING #########################
#################################################################

def maml_training(CONFIG):
    config_dict = vars(CONFIG)
    
    # ====== Create MAML Experiment Folder ======
    maml_folder = get_maml_folder(CONFIG)
    
    # ====== Load Dataset ======
    if CONFIG.maml_dataset == 'CelebA':
        # Load training and validation datasets for CelebA
        img_dataset = CelebA('train', data_root=CONFIG.data_root, max_len=CONFIG.max_len)
        val_img_dataset = CelebA('val', data_root=CONFIG.data_root, max_len=100)
        image_resolution = (img_dataset.size[1] // CONFIG.downscaling_factor, img_dataset.size[0] // CONFIG.downscaling_factor)
        coord_dataset = Implicit2DWrapper(img_dataset, sidelength=image_resolution)
        val_coord_dataset = Implicit2DWrapper(val_img_dataset, sidelength=image_resolution)
        img_channels = 3
    elif CONFIG.maml_dataset == 'DIV2K':
        # Load training and validation datasets for DIV2K
        img_dataset = DIV2K('train', data_root=CONFIG.data_root)
        val_img_dataset = DIV2K('val', data_root=CONFIG.data_root, max_len=100)
        image_resolution = (img_dataset.size[1] // CONFIG.downscaling_factor, img_dataset.size[0] // CONFIG.downscaling_factor)
        coord_dataset = Implicit2DWrapper(img_dataset, sidelength=image_resolution)
        val_coord_dataset = Implicit2DWrapper(val_img_dataset, sidelength=image_resolution)
        img_channels = 3
    else:
        print("Unknown dataset")
        return

    # Create DataLoaders for training and validation
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=CONFIG.maml_batch_size, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(val_coord_dataset, shuffle=True, batch_size=CONFIG.maml_batch_size, pin_memory=True, num_workers=0)

    # ====== Define the Model ======
    # Initialize an INRNet with the specified architecture and encoding
    model = INRNet(type=CONFIG.activation, mode=CONFIG.encoding, sidelength=image_resolution,
                   out_features=img_channels, hidden_features=CONFIG.hidden_dims,
                   num_hidden_layers=CONFIG.hidden_layers, encoding_scale=CONFIG.encoding_scale,
                   ff_dims=CONFIG.ff_dims).cuda()

    # Save the configuration to a YAML file for reproducibility
    yaml.dump(config_dict, open(os.path.join(maml_folder, 'CONFIG.yml'), 'w'))

    # Initialize the MAML model with the INRNet as the base model
    meta_siren = MAML(num_meta_steps=CONFIG.maml_adaptation_steps, hypo_module=model, loss=l2_loss,
                      init_lr=CONFIG.inner_lr).cuda()

    # Define optimizer and learning rate scheduler
    optim = torch.optim.Adam(lr=CONFIG.outer_lr, params=meta_siren.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, eps=1e-08,
                                                           verbose=True)

    # Initialize variables to track the best validation loss and state
    best_val_loss = float("Inf")
    best_state_dict = copy.deepcopy(meta_siren.state_dict())
    steps = CONFIG.maml_adaptation_steps
    step = 0

    # ====== Training Loop ======
    for i in range(CONFIG.maml_epochs):
        for model_input, gt in dataloader:
            step += 1
            # Prepare the sample for training
            sample = {'context': {'x': model_input['coords'], 'y': gt['img']},
                      'query': {'x': model_input['coords'], 'y': gt['img']}}
            sample = dict_to_gpu(sample)

            # Forward pass through the MAML model
            model_output = meta_siren(sample)
            loss = ((model_output['model_out'] - sample['query']['y']) ** 2).mean()

            # Log and visualize results at specified intervals
            if not step % CONFIG.steps_til_summary:
                visualized_steps = list(range(steps)[::int(steps / 3)])
                visualized_steps.append(steps - 1)
                print("Step %d, Total loss %0.6f" % (step, loss))
                fig, axes = plt.subplots(1, 5, figsize=(30, 6))
                ax_titles = ['Learned Initialization', 'Inner step {} output'.format(str(visualized_steps[0] + 1)),
                             'Inner step {} output'.format(str(visualized_steps[1] + 1)),
                             'Inner step {} output'.format(str(visualized_steps[2] + 1)),
                             'Ground Truth']
                for i, inner_step_out in enumerate([model_output['intermed_predictions'][i] for i in visualized_steps]):
                    plot_sample_image(inner_step_out, ax=axes[i], image_resolution=image_resolution)
                    axes[i].set_title(ax_titles[i], fontsize=25)
                plot_sample_image(model_output['model_out'], ax=axes[-2], image_resolution=image_resolution)
                axes[-2].set_title(ax_titles[-2], fontsize=25)

                plot_sample_image(sample['query']['y'], ax=axes[-1], image_resolution=image_resolution)
                axes[-1].set_title(ax_titles[-1], fontsize=25)
                plt.close(fig)

                # Save model checkpoints
                torch.save(model.state_dict(), os.path.join(maml_folder, 'model_maml_step{}.pth'.format(step)))
                torch.save(meta_siren.state_dict(), os.path.join(maml_folder, 'maml_obj_step{}.pth'.format(step)))

            # Backward pass and optimization step
            optim.zero_grad()
            loss.backward()
            optim.step()

            # ====== Validation ======
            if not step % 500:
                val_loss_sum = 0
                meta_siren.first_order = True
                with torch.no_grad():
                    for val_step, (model_input, gt) in enumerate(val_dataloader):
                        sample = {'context': {'x': model_input['coords'], 'y': gt['img']},
                                  'query': {'x': model_input['coords'], 'y': gt['img']}}
                        sample = dict_to_gpu(sample)
                        model_output = meta_siren(sample)
                        val_loss = ((model_output['model_out'] - sample['query']['y']) ** 2).mean().detach().cpu()
                        val_loss_sum += val_loss

                    print("Validation loss: ", val_loss_sum.item())
                    if val_loss_sum < best_val_loss:
                        best_state_dict = copy.deepcopy(meta_siren.state_dict())
                        best_val_loss = val_loss_sum
                    scheduler.step(val_loss_sum)
                meta_siren.first_order = False

    # Load the best model state
    meta_siren.load_state_dict(best_state_dict)
    model = meta_siren.hypo_module

    # Save the final model and MAML object
    torch.save(model.state_dict(), os.path.join(maml_folder, 'model_maml.pth'))
    torch.save(meta_siren.state_dict(), os.path.join(maml_folder, 'maml_obj.pth'))



################################################################################
######################### MAML OVERFITTING EXPERIMENT ##########################
################################################################################

def refine(maml, context_dict, steps):
    """Specializes the model"""
    x = context_dict.get('x').cuda()
    y = context_dict.get('y').cuda()

    meta_batch_size = x.shape[0]
    with torch.enable_grad():
        fast_params = OrderedDict()
        for name, param in maml.hypo_module.meta_named_parameters():
            fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))
        prev_loss = 1e6
        intermed_predictions = []
        for j in range(steps):
            # Using the current set of parameters, perform a forward pass with the context inputs.
            predictions = maml.hypo_module({'coords': x}, params=fast_params)
            # Compute the loss on the context labels.
            loss = maml.loss(predictions, y)
            intermed_predictions.append(predictions['model_out'].detach().cpu())
            fast_params, grads = maml._update_step(loss, fast_params, j)
            prev_loss = loss

    return fast_params, intermed_predictions


def maml_overfitting_experiment(CONFIG):
    config_dict = vars(CONFIG)

    # ====== Load Image and Experiment Paths ======
    image_paths = glob.glob(os.path.join(CONFIG.data_root, CONFIG.dataset, '*'))
    experiment_folder = get_maml_overfitting_experiment_folder(CONFIG)

    # ====== Prepare MAML Configuration and Model ======
    maml_folder = get_maml_folder(CONFIG)
    MAML_CONFIG = yaml.load(open(os.path.join(maml_folder, 'CONFIG.yml'), 'r'))
    yaml.dump(config_dict, open(os.path.join(experiment_folder, 'CONFIG.yml'), 'w'))
    maml_state_dict = torch.load(os.path.join(maml_folder, 'model_maml.pth'), map_location='cpu') # Load the pre-trained MAML model's state dictionary (parameters).
    torch.save(maml_state_dict, os.path.join(experiment_folder, 'model_maml.pth')) #  Save the state dictionary to the experiment folder for reference

    # ====== Process Each Image : Create one model for each image and store the weights ======
    for i, im in enumerate(image_paths):
        image_name = im.split('/')[-1].split('.')[0]

        # Flip vertically oriented Kodak images to use the same orientation for all images
        if img.img.size[1] > img.img.size[0] and CONFIG.dataset == 'KODAK':
            img.img = img.img.rotate(90, expand=1)

        # ====== Create DataLoader ======
        # Convert image into a dataset of coordinate-value pairs
        img = ImageFile(im)
        img_dataset = img
        image_resolution = (img.img.size[1] // CONFIG.downscaling_factor, img.img.size[0] // CONFIG.downscaling_factor)
        # Convert it into a coordinate dataset
        coord_dataset = Implicit2DWrapper(img_dataset, sidelength=image_resolution)
        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=CONFIG.maml_batch_size, pin_memory=True, num_workers=0)

        # ====== Define the Model ======
        # Initialize an INRNet with the specified architecture and encoding
        model = INRNet(type=CONFIG.activation, mode=CONFIG.encoding, sidelength=image_resolution,
                                out_features=img_dataset.img_channels, hidden_features=CONFIG.hidden_dims,
                                num_hidden_layers=CONFIG.hidden_layers, encoding_scale=CONFIG.encoding_scale,
                                ff_dims=CONFIG.ff_dims)

        root_path = os.path.join(experiment_folder, image_name)
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if os.path.exists(os.path.join(root_path, 'checkpoints', 'model_final.pth')):
            # result exists already -> skip
            print("Skipping ", root_path)
            continue

        # ====== Load the MAML Model ======
        # Initialize a MAML model using the loaded INR model and meta-learning settings.
        meta_siren = MAML(num_meta_steps=MAML_CONFIG['maml_adaptation_steps'], hypo_module=model,
                            loss=l2_loss, init_lr=MAML_CONFIG['inner_lr'],
                            lr_type=MAML_CONFIG['lr_type']).cuda()
        
        # Load the pre-trained MAML model's parameters into the meta_siren.
        state_dict = torch.load(os.path.join(maml_folder, 'maml_obj.pth'), map_location='cpu')
        meta_siren.load_state_dict(state_dict, strict=True)

        # Set the model to first-order mode (no second-order gradients) for faster adaptation.
        meta_siren.first_order = True
        eval_model = copy.deepcopy(meta_siren.hypo_module)

        # Store the number of MAML adaptation steps.
        num_maml_steps = CONFIG.maml_adaptation_steps


        # ====== Adapt the Model to Each Pixel in the image ======
        for step, (model_input, ground_truth) in enumerate(dataloader):
            # Context/Query contain pixel coordinates, y the corresponding RGB value
            sample = {'context': {'x': model_input['coords'], 'y': ground_truth['img']},
                        'query': {'x': model_input['coords'], 'y': ground_truth['img']}}
            sample = dict_to_gpu(sample)
            context = sample['context']
            query_x = sample['query']['x'].cuda()

            # ====== Meta-Learning Adaptation (Refine the Model) ======
            # Adapt the model to the current image using the "refine" function.
            # This performs a few gradient steps starting from the meta-learned initialization.
            fast_params, intermed_predictions = refine(meta_siren, context, num_maml_steps)

            # ====== Compute Final Outputs ======
            # Use the adapted parameters to generate predictions for the query set.
            model_output = meta_siren.hypo_module({'coords': query_x}, params=fast_params)['model_out']
            model_output = {'model_out': model_output, 'intermed_predictions': intermed_predictions,
                            'fast_params': fast_params}
            
            # ====== Setup Loss and Evaluation Functions ======
            loss_fn = image_mse
            summary_fn = partial(write_image_summary, image_resolution)
            fast_params_squeezed = {name: param.squeeze() for name, param in fast_params.items()}
            eval_model.load_state_dict(fast_params_squeezed)

            print('Metrics after 3 steps: ', compute_metrics(model_output['model_out'], sample['query']['y'], dataloader.dataset.sidelength))

            l1_loss_fn = partial(model_l1_diff, meta_siren.hypo_module)

            # ====== Fine-Tune the Model (Overfitting) ======
            # Continue training (fine-tuning) the adapted model using standard optimization.
            training_utils.train(model=eval_model, train_dataloader=dataloader, epochs=CONFIG.epochs, lr=CONFIG.lr,
                            steps_til_summary=CONFIG.steps_til_summary, epochs_til_checkpoint=CONFIG.epochs_til_ckpt,
                            model_dir=root_path, loss_fn=loss_fn, l1_loss_fn=l1_loss_fn, summary_fn=summary_fn,
                            l1_reg=CONFIG.l1_reg, patience=CONFIG.patience, warmup=CONFIG.warmup)

            metrics = check_metrics_full(dataloader, eval_model, image_resolution)
            metrics = {'mse': metrics[0], 'psnr': metrics[2], 'ssim': metrics[1]}

            with open(os.path.join(root_path, 'result.json'), 'w') as fp:
                json.dump(metrics, fp)


