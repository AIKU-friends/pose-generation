import os
import tarfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SitcomPoseDataset(Dataset):
    def __init__(self, data_path, data_list):

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_path = data_path
        self.image_dir_path = os.path.join(self.data_path, 'data')
        self.annotation_list = self.preprocess(data_list)
        
    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        target_point, image_name, pose_keypoints, one_hot_encoded_pose_cluster, scale, deformation = self.annotation_list[idx]

        image = Image.open(os.path.join(self.image_dir_path, image_name))
        width, height = image.size

        crop_box = ((target_point[0] - (height // 2)), (target_point[1] - (height // 2)), (target_point[0] + (height // 2)), (target_point[1] + (height // 2)))
        zoom_box = ((target_point[0] - (height // 4)), (target_point[1] - (height // 4)), (target_point[0] + (height // 4)), (target_point[1] + (height // 4)))
        image_crop = image.crop(crop_box)
        image_zoom = image.crop(zoom_box)

        deformation_list = [item for sublist in deformation for item in sublist]
        scale_deformation_list = scale + deformation_list

        transformed_image = self.transform(image)
        transformed_image_crop = self.transform(image_crop)
        transformed_image_zoom = self.transform(image_zoom)
        
        one_hot_encoded_pose_cluster = torch.tensor(one_hot_encoded_pose_cluster, dtype=torch.float32)
        scale_deformation_list = torch.tensor(scale_deformation_list)

        return transformed_image, transformed_image_crop, transformed_image_zoom, one_hot_encoded_pose_cluster, scale_deformation_list
        
    def preprocess(self, data_list):

        cluster_keypoints_list = []
        with open(os.path.join(self.data_path, 'centers_30.txt'), 'r') as f:
            cluster_data_list = list(f.readlines())
        for cluster_data in cluster_data_list:
            cluster_data = cluster_data.split(' ')[:-1]
            cluster_data = [float(x) for x in cluster_data]
            cluster_keypoints = []
            for i in range(0, len(cluster_data), 2):
                cluster_keypoints.append((cluster_data[i], cluster_data[i+1]))
            cluster_keypoints[:-2]
            cluster_keypoints_list.append(cluster_keypoints)
        
        annotation_list = []
        
        for data in data_list:
            splited_data = data.split(' ')

            image_name = splited_data[0]
            pose_data = splited_data[1:-1]
            pose_data = [int(eval(x)) for x in pose_data]
            pose_keypoints = []
            for i in range(0, len(pose_data), 2):
                pose_keypoints.append((pose_data[i], pose_data[i+1]))
            target_point = pose_keypoints[-1:][0]
            pose_keypoints = pose_keypoints[:-1]

            # target_point recalculate (mean)
            target_point = (sum([x[0] for x in pose_keypoints]) / len(pose_keypoints), sum([x[1] for x in pose_keypoints]) / len(pose_keypoints))

            pose_cluster = eval(splited_data[-1]) - 1
            one_hot_encoded_pose_cluster = self.one_hot_encode(pose_cluster, len(cluster_keypoints_list))
            
            scale = self.cal_scale(cluster_keypoints_list[pose_cluster], pose_keypoints)
            deformation = self.cal_deformation(cluster_keypoints, pose_keypoints, scale, target_point)

            annotation = (target_point, image_name, pose_keypoints, one_hot_encoded_pose_cluster, scale, deformation)
            annotation_list.append(annotation)

            scaled_cluster = [(point[0] * scale[0] + target_point[0], point[1] * scale[1] + target_point[1]) for point in cluster_keypoints_list[pose_cluster]]
            predicted = [(x[0] + y[0], x[1] + y[1]) for x, y in zip(deformation, scaled_cluster)]
        
        return annotation_list

    def cal_scale(self, cluster_keypoints, target_keypoints):
        cluster_x = [point[0] for point in cluster_keypoints]
        cluster_y = [point[1] for point in cluster_keypoints]
        target_x = [point[0] for point in target_keypoints]
        target_y = [point[1] for point in target_keypoints]

        s_x = (max(target_x) - min(target_x)) / (max(cluster_x) - min(cluster_x))
        s_y = (max(target_y) - min(target_y)) / (max(cluster_y) - min(cluster_y))
        
        return [s_x, s_y]
    
    def cal_deformation(self, cluster_keypoints, target_keypoints, scale, target_point):
        scaled_cluster = [(point[0] * scale[0] + target_point[0], point[1] * scale[1] + target_point[1]) for point in cluster_keypoints]
        deformation = [(x[0] - y[0], x[1] - y[1]) for x, y in zip(target_keypoints, scaled_cluster)]

        return deformation
    
    def one_hot_encode(self, value, num_classes):
        vector = [0] * num_classes
        vector[value] = 1

        return vector


if __name__ == '__main__':
    data_path = './affordance_data'
    data_list = []
    with open(os.path.join(data_path, 'trainlist.txt'), 'r') as f:
        data_list = list(f.readlines())

    dataset = SitcomPoseDataset(data_path, data_list)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for data in dataloader:
        print(data)
        break