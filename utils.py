import torch

def generate_pose(base_pose, scale, deformation, target_point, cfg):
    target_point_method = cfg.target_point_method
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