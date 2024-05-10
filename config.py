import os

class Config:
    def __init__(self):
        
        self.alexnet_fc7_dim = 4096
        self.pose_dim = 14

        self.fc_dim = 512
        self.latent_dim = 30
        self.sd_dim = 34
        self.class_weight = [0.6072, 0.32255, 1.45106, 1.69682, 2.15381, 0.6999, 1.21317, 1.09274, 2.76061, 0.79155, 1.97931, 1.76047, 1.08609, 1.38459]

        self.target_point_method = 'mean' # ['mean', 'center', keypoint_index]
        self.data_tag = '_kmeans_2'

        self.num_epochs = 1000
        self.validation_term = 1
        self.variational_beta = 5
        self.learning_rate = 4e-4
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.weight_decay = 2e-5
        self.CLIP = 1
        self.batch_size =  256
        
        self.pck_threshold = 0.2

        self.backbone_freeze = False

        self.checkpoint_dir = 'checkpoints/experiment26'
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)