import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import SitcomPoseDataset
from model import VariationalAutoencoder, PoseClassifier
from train_vae import generate_pose, cal_MSE, cal_PCK

device = 'cuda:0'
use_classifier = True
use_encoder = False

# k=30
classifier_model_path = 'checkpoints/experiment31/classifier_sota.pt'
vae_model_path = 'checkpoints/experiment31/model_70_813_706.pt'

# k=14
# classifier_model_path = 'checkpoints/experiment33/model_4_3_[0.14418603479862213, 0.2643410861492157, 0.3852713108062744, 0.5023255348205566, 0.593281626701355].pt'
# vae_model_path = 'checkpoints/experiment33/model_41_959_736.pt'

# k=14 (filtered)
# classifier_model_path = 'checkpoints/experiment28/model_46_6_[0.1723514199256897, 0.3007751703262329, 0.41111108660697937, 0.5054263472557068, 0.5813953280448914].pt'
# vae_model_path = 'checkpoints/experiment28/model_245_2086_2084.pt'

cfg = Config()
model = VariationalAutoencoder(cfg).to(device)
model.load_state_dict(torch.load(vae_model_path, map_location=device))

if use_classifier:
    classifer = PoseClassifier(cfg).to(device)
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
            pose_prob = classifer(img.to(device), img_crop.to(device), img_zoom.to(device))
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