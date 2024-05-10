import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import SitcomPoseDataset
from model import VariationalAutoencoder, PoseClassifier
from train_vae import generate_pose, cal_MSE, cal_PCK

device = 'cpu'
use_classifier = True
use_encoder = False
classifier_model_path = 'checkpoints/experiment22/model_81_1720_1720.pt'
vae_model_path = 'checkpoints/experiment22/model_81_1720_1720.pt'

cfg = Config()
model = VariationalAutoencoder(cfg)
model.load_state_dict(torch.load(vae_model_path, map_location=device))

if use_classifier:
    classifer = PoseClassifier(cfg)
    classifer.load_state_dict(torch.load(classifier_model_path, map_location=device))
    classifer.eval()

model.eval()

data_path = './affordance_data'
data_list = []

with open(os.path.join(data_path, f'testlist{cfg.data_tag}.txt'), 'r') as f:
    data_list = list(f.readlines())
test_dataset = SitcomPoseDataset(data_path, data_list, cfg)
dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

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


mse_list = []
pck_list = []

with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader), total=dataloader.__len__()):
        
        img, img_crop, img_zoom, one_hot_pose_vector, scale_deformation, pose_keypoints, image_size, target_point = batch

        if use_classifier:
            pose_prob = classifer(img, img_crop, img_zoom)
            pose_cluster = torch.argmax(pose_prob, dim=1)
            one_hot_pose_vector = torch.nn.functional.one_hot(pose_cluster, one_hot_pose_vector.size(1))
            one_hot_pose_vector = torch.tensor(one_hot_pose_vector, dtype=torch.float32)
        else:
            pose_cluster = torch.argmax(one_hot_pose_vector, dim=1)

        base_pose = torch.stack([cluster_keypoints_list[value] for value in pose_cluster])

        if use_encoder:
            scale_deformation_recon, latent_mu, latent_logvar = model(scale_deformation.to(device), one_hot_pose_vector.to(device), img.to(device), img_crop.to(device), img_zoom.to(device))
        else:
            latent_vector = torch.randn((base_pose.shape[0], cfg.latent_dim))
            scale_deformation_recon = model.decoder(latent_vector.to(device), one_hot_pose_vector.to(device), img.to(device), img_crop.to(device), img_zoom.to(device))

        generated_pose = generate_pose(base_pose, scale_deformation_recon[:, :2].to('cpu'), scale_deformation_recon[:, 2:].to('cpu'), target_point)
        mse = cal_MSE(generated_pose, pose_keypoints)
        pck = cal_PCK(generated_pose, pose_keypoints, cfg.pck_threshold)

        mse_list.append(mse)
        pck_list.append(pck)

ave_pck = np.mean(pck_list)
ave_mse = np.mean(mse_list)

print(ave_pck, ave_mse)