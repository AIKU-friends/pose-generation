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
        self.pose_dim = 30

        self.fc_dim = 512
        self.latent_dim = 30
        self.sd_dim = 34
        self.class_weight = [2.74664, 0.99431, 1.12161, 1.60353, 1.97212, 0.97565, 1.65784, 1.85766, 1.24029, 1.90442, 1.1788, 1.46778, 0.63529, 0.58117, 2.16727, 1.46006, 0.43825, 1.11113, 0.38999, 1.14633, 0.79716, 0.80799, 2.01998, 1.51867, 1.08789, 0.97565, 0.47692, 1.44736, 1.63183, 0.65324]
        self.target_point_method = 'mean' # ['mean', 'center', keypoint_index]
        self.data_tag = '_kmeans_1'

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

        self.checkpoint_dir = 'checkpoints/experiment31'
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)