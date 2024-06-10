import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import Config

class AlexNetFc7(nn.Module):
    def __init__(self, freeze):
        super(AlexNetFc7, self).__init__()
        model = models.alexnet(weights='IMAGENET1K_V1')
        modules = list(model.children())
        modules.insert(-1, nn.Flatten())
        modules[-1] = modules[-1][:-1]

        self.alexnet = nn.Sequential(*modules)
        self.freeze = freeze

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.alexnet(x)
        else:
            x = self.alexnet(x)
        return x
    
class VGG16_fc(nn.Module):
    def __init__(self, freeze):
        super(VGG16_fc, self).__init__()
        modules = list(models.vgg16(weights='IMAGENET1K_V1').children())[:-1]
        modules.append(nn.Flatten())

        self.vgg16 = nn.Sequential(*modules)
        self.freeze = freeze

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.vgg16(x)
        else:
            x = self.vgg16(x)

        return x
    
class ResNet18_Upsampling(nn.Module):
    def __init__(self, freeze):
        super(ResNet18_Upsampling, self).__init__()
        model = models.resnet18(weights='IMAGENET1K_V1')
        modules = list(model.children())
        self.sublayer1 = nn.Sequential(*modules[:-4])
        self.sublayer2 = nn.Sequential(*modules[-4])
        self.sublayer3 = nn.Sequential(*modules[-3])
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
        self.projection_layer1 = nn.Conv2d(256, 512, 1)
        self.projection_layer2 = nn.Conv2d(128, 512, 1)
        self.projection_layer3 = nn.Conv2d(512, 128, 1)
        self.flatten = nn.Flatten()
        self.freeze = freeze

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                feature1 = self.sublayer1(x)
                feature2 = self.sublayer2(feature1)
                feature3 = self.sublayer3(feature2)
        else:
            feature1 = self.sublayer1(x)
            feature2 = self.sublayer2(feature1)
            feature3 = self.sublayer3(feature2)
        
        x = self.upsample_layer(feature3) + self.projection_layer1(feature2)
        x = self.upsample_layer(x) + self.projection_layer2(feature1)
        x = self.flatten(self.projection_layer3(x))
        return x
    
class ConditionEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConditionEncoder, self).__init__()

        if cfg.backbone == 'AlexNet':
            self.backbone = AlexNetFc7(cfg.backbone_freeze)
        if cfg.backbone == 'VGG':
            self.backbone = VGG16_fc(cfg.backbone_freeze)
        if cfg.backbone == 'ResNetUp':
            self.backbone = ResNet18_Upsampling(cfg.backbone_freeze)

        self.fc1 = nn.Linear(in_features=cfg.pose_dim, out_features=cfg.fc_dim)
        self.fc2 = nn.Linear(in_features=cfg.fc_dim + 3*cfg.backbone_feature_dim, out_features=cfg.fc_dim)

    def forward(self, pose, img, img_crop, img_zoom):
        pose = F.relu(self.fc1(pose))
        img = self.backbone(img)
        img_crop = self.backbone(img_crop)
        img_zoom = self.backbone(img_zoom)

        output = torch.cat((pose, img, img_crop, img_zoom), dim=1)
        output = F.relu(self.fc2(output))

        return output

class Encoder(nn.Module):
    def __init__(self, cfg, condition_encoder):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=cfg.sd_dim, out_features=cfg.fc_dim)
        self.fc2 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)

        self.condition_encoder = condition_encoder
        self.fc_latent = nn.Linear(in_features=2 * cfg.fc_dim, out_features=cfg.latent_dim)

    def forward(self, x, pose, img, img_crop, img_zoom):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        c = self.condition_encoder(pose, img, img_crop, img_zoom)

        x = torch.cat((x, c), dim=1)
        x = self.fc_latent(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, cfg, condition_encoder):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=cfg.latent_dim, out_features=cfg.fc_dim)
        self.fc2 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)

        self.condition_encoder = condition_encoder

        self.fc3 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)
        self.fc4 = nn.Linear(in_features=2 * cfg.fc_dim, out_features=cfg.fc_dim)
        self.fc5 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.fc_dim)
        self.fc6 = nn.Linear(in_features=cfg.fc_dim, out_features=cfg.sd_dim)

    def forward(self, x, pose, img, img_crop, img_zoom):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        c = self.condition_encoder(pose, img, img_crop, img_zoom)
        c = F.relu(self.fc3(c))

        x = torch.cat((x, c), dim=1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.0, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings

    
class VQ_VAE(nn.Module):
    def __init__(self, cfg):
        super(VQ_VAE, self).__init__()
        self.condition_encoder = ConditionEncoder(cfg)
        self.encoder = Encoder(cfg, self.condition_encoder)
        self.decoder = Decoder(cfg, self.condition_encoder)
        self.vq = VectorQuantizerEMA(cfg.num_embeddings, cfg.latent_dim, cfg.commitment_cost, cfg.decay)

    def forward(self, x, pose, img, img_crop, img_zoom):
        latent = self.encoder(x, pose, img, img_crop, img_zoom)
        loss, quantized, perplexity, _ = self.vq(latent)
        x_recon = self.decoder(quantized, pose, img, img_crop, img_zoom)
        
        return loss, x_recon, perplexity
    
    def inference(self, pose, img, img_crop, img_zoom):
        codebook = self.vq._embedding
        sampled_indices = torch.randint(0, codebook.weight.size(0), (img.size(0),))
        quantized_latents = codebook.weight[sampled_indices]
        outputs = self.decoder(quantized_latents, pose, img, img_crop, img_zoom)

        return outputs
        
if __name__ == '__main__':
    cfg = Config()
    vq_vae = VQ_VAE(cfg)
    batch_size = 2
    x = torch.randn((batch_size, cfg.sd_dim))
    pose = torch.randn((batch_size, cfg.pose_dim))
    img1 = torch.randn((batch_size, 3, 224, 224))
    img2 = torch.randn((batch_size, 3, 224, 224))
    img3 = torch.randn((batch_size, 3, 224, 224))
    vq_vae.train()
    loss, x_recon, perplexity = vq_vae(x, pose, img1, img2, img3)
    print(perplexity)
    print("===VQ_VAE OUTPUT===")
    print(x_recon.shape)

    print("===DECODER OUTPUT===")
    x_recon = vq_vae.inference(pose, img1, img2, img3)
    print(x_recon.shape)

    print("===Parmeter Count===")
    num_params = sum(p.numel() for p in vq_vae.parameters() if p.requires_grad)
    print(num_params)