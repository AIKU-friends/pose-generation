import os

class Config:
    def __init__(self):
        
        self.backbone = 'VGG' # ['AlexNet', 'VGG', 'ResNetUp']
        if self.backbone == 'AlexNet':
            self.backbone_feature_dim = 4096
        if self.backbone == 'VGG':
            self.backbone_feature_dim = 25088
        if self.backbone == 'ResNetUp':
            self.backbone_feature_dim = 100352
        self.pose_dim = 14

        self.fc_dim = 512
        self.latent_dim = 30
        self.sd_dim = 34
        self.class_weight = [0.6072, 0.32255, 1.45106, 1.69682, 2.15381, 0.6999, 1.21317, 1.09274, 2.76061, 0.79155, 1.97931, 1.76047, 1.08609, 1.38459]
        self.target_point_method = 'mean' # ['mean', 'center', keypoint_index]
        self.data_tag = '_kmeans_2'

        self.num_epochs = 1000
        self.validation_term = 1
        self.variational_beta = 10
        self.learning_rate = 2e-4
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.weight_decay = 2e-5
        self.CLIP = 1
        self.batch_size = 16
        
        self.pck_threshold = 0.2

        self.backbone_freeze = False

        self.checkpoint_dir = 'checkpoints/experiment33'
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)