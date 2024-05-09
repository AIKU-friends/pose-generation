import math
import os

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from config import Config
from data_loader import SitcomPoseDataset
from model import VariationalAutoencoder

device = 'cuda:0'

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def train_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_dataloader, clip):
    ## args
    # - criterion: 학습 손실을 계산하는 데 사용되는 손실 함수
    # - clip: for gradient clipping, prevent explosion

    losses = []
    model.train()

    for i, batch in enumerate(train_dataloader):

        ## -- forward step
        img, img_crop, img_zoom, one_hot_pose_vector, scale_deformation, _, _, _ = batch

        # make model output - vae
        scale_deformation_recon, latent_mu, latent_logvar = model(scale_deformation.to(device), one_hot_pose_vector.to(device), img.to(device), img_crop.to(device), img_zoom.to(device))

        # compute loss using loss function
        loss, _ = criterion(scale_deformation_recon.to(device), scale_deformation.to(device), latent_mu.to(device), latent_logvar.to(device))

        loss_val = loss.item()
        losses.append(loss_val)

        wandb.log({"Train Loss": math.log10(loss_val)})

        ## -- optimization step
        optimizer.zero_grad()
        # gradient 계산
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer - update model parameter
        optimizer.step()
        # update the learning rate
        # lr_scheduler.step()

    return np.mean(losses)

def eval_epoch(model, dataloader, criterion, cluster_keypoints_list, pck_threshold, latent_dim):
    # - criterion: 검증 손실을 계산하는 데 사용되는 손실 함수
    model.eval()
    val_losses = []
    mse_losses = []
    mse_list = []
    pck_list = []
    mse_latent_list = []
    pck_latent_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            
            ## -- forward step
            img, img_crop, img_zoom, one_hot_pose_vector, scale_deformation, pose_keypoints, image_size, target_point = batch

            # make model output - vae
            scale_deformation_recon, latent_mu, latent_logvar = model(scale_deformation.to(device), one_hot_pose_vector.to(device), img.to(device), img_crop.to(device), img_zoom.to(device))

            # compute loss using loss function
            loss, mse_loss = criterion(scale_deformation_recon.to(device), scale_deformation.to(device), latent_mu.to(device), latent_logvar.to(device))

            loss_val = loss.item()
            val_losses.append(loss_val)
            mse_losses.append(mse_loss.item())


            pose_cluster = torch.argmax(one_hot_pose_vector, dim=1)
            base_pose = torch.stack([cluster_keypoints_list[value] for value in pose_cluster])

            generated_pose = generate_pose(base_pose, scale_deformation_recon[:, :2].to('cpu'), scale_deformation_recon[:, 2:].to('cpu'), target_point)
            mse = cal_MSE(generated_pose, pose_keypoints)
            pck = cal_PCK(generated_pose, pose_keypoints, pck_threshold)
            mse_list.append(mse)
            pck_list.append(pck)

            latent_vector = torch.randn((scale_deformation.shape[0], latent_dim))
            scale_deformation_recon = model.decoder(latent_vector.to(device), one_hot_pose_vector.to(device), img.to(device), img_crop.to(device), img_zoom.to(device))

            generated_pose = generate_pose(base_pose, scale_deformation_recon[:, :2].to('cpu'), scale_deformation_recon[:, 2:].to('cpu'), target_point)
            mse = cal_MSE(generated_pose, pose_keypoints)
            pck = cal_PCK(generated_pose, pose_keypoints, pck_threshold)
            mse_latent_list.append(mse)
            pck_latent_list.append(pck)

    return np.mean(val_losses), np.mean(mse_losses), np.mean(mse_list), np.mean(pck_list), np.mean(mse_latent_list), np.mean(pck_latent_list)

def train():

    cfg = Config()
    model = VariationalAutoencoder(cfg).to(device)
    # model.load_state_dict(torch.load('checkpoints/experiment12/model_63_1929_1433.pt'))

    def vae_loss(recon_x, x, mu, logvar):

        recon_loss = F.mse_loss(recon_x.view(-1, cfg.sd_dim), x.view(-1, cfg.sd_dim), reduction='mean')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + cfg.variational_beta * kldivergence, recon_loss

    criterion = vae_loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = cfg.num_epochs

    data_path = './affordance_data'
    data_list = []

    with open(os.path.join(data_path, f'trainlist{cfg.data_tag}.txt'), 'r') as f:
        data_list = list(f.readlines())
    train_dataset = SitcomPoseDataset(data_path, data_list, cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    with open(os.path.join(data_path, f'testlist{cfg.data_tag}.txt'), 'r') as f:
        data_list = list(f.readlines())
    test_dataset = SitcomPoseDataset(data_path, data_list, cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    cluster_keypoints_list = []
    with open(os.path.join(data_path, f'centers_30{cfg.data_tag}.txt'), 'r') as f:
        cluster_data_list = list(f.readlines())
    for cluster_data in cluster_data_list:
        cluster_data = cluster_data.split(' ')[:-1]
        cluster_data = [float(x) for x in cluster_data]
        cluster_keypoints = []
        for i in range(0, len(cluster_data), 2):
            cluster_keypoints.append((cluster_data[i], cluster_data[i+1]))
        cluster_keypoints = cluster_keypoints[:-1]
        cluster_keypoints_list.append(torch.tensor(cluster_keypoints))
    cluster_keypoints_list = torch.stack(cluster_keypoints_list)

    train_loss_avg = []
    print("Training ...")

    for epoch in range(num_epochs):

        current_lr = get_lr(optimizer)
        print(f'Epoch {epoch+1}/{num_epochs}, current lr={current_lr}')

        train_loss = train_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, cfg.CLIP)
        train_loss_avg.append(train_loss)
        # lr_scheduler.step()

        # validation step
        if num_epochs % cfg.validation_term == 0 and num_epochs != 0:
            valid_loss, mse_loss, mse, pck, mse_l, pck_l = eval_epoch(model, test_loader, criterion, cluster_keypoints_list, cfg.pck_threshold, cfg.latent_dim)
            wandb.log({"Validation Loss": valid_loss, "MSE Loss": mse_loss, "MSE": mse, f"PCK@{cfg.pck_threshold}": pck, "MSE_Latent": mse_l, f"PCK@{cfg.pck_threshold}_Latent": pck_l})
            print(f'Validation Loss: {valid_loss:.3f}, MSE Loss: {mse_loss:.3f}, MSE: {mse:.3f}, PCK@{cfg.pck_threshold}: {pck:.3f}, MSE_Latent: {mse_l:.3f}, PCK@{cfg.pck_threshold}_Latent: {pck_l:.3f}')

            # validation 후 - model save
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f'model_{epoch}_{int(valid_loss)}_{int(mse_loss)}.pt'))

def generate_pose(base_pose, scale, deformation, target_point):
    target_point_method = Config().target_point_method
    scaled_cluster = base_pose * scale.unsqueeze(1)
    deformed_pose = scaled_cluster + deformation.view(-1, 16, 2)
    if target_point_method == 'mean':
        center_position = torch.mean(deformed_pose, dim=1)
    if target_point_method == 'center':
        max_values, _ = torch.max(deformed_pose, dim=1)
        min_values, _ = torch.min(deformed_pose, dim=1)
        center_position = torch.stack([(max_values[:, 0] + min_values[:, 0]) / 2, (max_values[:, 1] + min_values[:, 1]) / 2], dim=1)
    if type(target_point_method) == int:
        center_position = deformed_pose[:, target_point_method, :]
    generated_pose = deformed_pose + (target_point - center_position).unsqueeze(1)

    return generated_pose

def cal_PCK(pred, target, threshold):
    num_keypoints = target.shape[1]

    distance = torch.norm(pred - target, dim=-1)
    distance_threshold = torch.norm(target[:, 3, :] - target[:, 12, :], dim=1).unsqueeze(1) * threshold
    pck_per_batch = (distance <= distance_threshold).sum(dim=-1) / num_keypoints
    pck = torch.mean(pck_per_batch, dim=0)
    return pck

def cal_MSE(pred, target):
    mse = torch.sum((pred - target).pow(2)) / (pred.shape[0] * target.shape[1] * 2)
    return mse

if __name__ == '__main__':
    experiment_config = Config()
    wandb.init(
        project="AIKU-VAE2",
        config=experiment_config.__dict__
    )
    train()