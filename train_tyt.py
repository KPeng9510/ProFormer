# The testing module requires faiss
from functools import partial
import sys
# So if you don't have that, then this import will break
from pml import trainers
from pml import losses, miners, samplers, testers, utils
#import losss
import torch.nn as nn
from vit_pytorch.swin import build_model
import record_keeper
import sklearn
from utils import common_functions as c_f
import pml.utils.logging_presets as logging_presets
import pml
import pml as pytorch_metric_learning
from torchvision import datasets, models, transforms
import torchvision
import logging
logging.getLogger().setLevel(logging.INFO)
import os
#from pytorch_pretrained_vit import ViT
from pml.losses.base_metric_loss_function import BaseMetricLossFunction
from pml.testers.base_tester import BaseTester
from vit_pytorch.pvt import PyramidVisionTransformer
from vit_pytorch.CausalLevit import LeViT_384
logging.info("pytorch-metric-learning VERSION %s"%pytorch_metric_learning.__version__)
logging.info("record_keeper VERSION %s"%record_keeper.__version__)
import logging
from sklearn.metrics import accuracy_score
from vit_pytorch.ResT import rest_small
#from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
import pickle

import hydra
from omegaconf import DictConfig


# reprodcibile
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class Loss_n(BaseMetricLossFunction):
    def __init__(self):
        super().__init__()
        self.n_pair_loss = losses.NPairsLoss()
        self.angular_loss = losses.AngularLoss(alpha=45)
    def compute_loss(self, embeddings, labels, indices_tuple):
        dict_angular = self.angular_loss.compute_loss(embeddings, labels, indices_tuple)
        dict_npair = self.n_pair_loss.compute_loss(embeddings, labels, indices_tuple)
        losses = 0.01*dict_angular['loss']['losses']+0.1*dict_npair['loss']['losses']
        dict_angular['loss']['losses']=losses
        return dict_angular
def calibration_augmentation(base_means, base_cov,embedding_and_labels):
    n_shot = 1
    n_ways = 20
    support_data = np.nan_to_num(embedding_and_labels['samples'][0])
    support_label = embedding_and_labels['samples'][1]
    sampled_data = []
    sampled_label = []
    num_sampled = int(10/n_shot)
    for i in range(20):
        mean, cov = distribution_calibration(support_data[np.squeeze(support_label == i),:], base_means, base_cov, k=4)
        sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
        sampled_label.extend([support_label[i]]*num_sampled)
    sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
    X_aug = np.concatenate([support_data, sampled_data])
    Y_aug = np.concatenate([support_label, sampled_label])
    return X_aug,Y_aug
def distribution_calibration( query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query - base_means[i]))
    index = np.argpartition(dist, k)[:k]

    mean = np.concatenate([np.array(base_means)[index], query])
    calibrated_mean = np.mean(mean, axis=0)
    #print(base_cov)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + alpha

    return calibrated_mean, calibrated_cov
class OneShotTester(BaseTester):

    def __init__(self, end_of_testing_hook=None):
        super().__init__()
        self.max_accuracy = 0.0
        self.embedding_filename = ""
        self.end_of_testing_hook = end_of_testing_hook


    def __get_correct(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            #             print(correct)
        return correct


    def __accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            correct = self.__get_correct(output, target, topk)
            batch_size = target.size(0)
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res



    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
        # print(embeet dings_and_labels)
        # train_embeddings = embeddings_and_labels['train'][0]
        # train_labels = embeddings_and_labels['train'][1]
        # print(train_embeddings.shape)
        # print(train_labels.shape)
        query_embeddings = embeddings_and_labels["val"][0]
        query_labels = embeddings_and_labels["val"][1]
        query_embeddings_2 = embeddings_and_labels["val_2"][0]
        query_labels_2 = embeddings_and_labels["val_2"][1]
        query_embeddings_3 = embeddings_and_labels["val_3"][0]
        query_labels_3 = embeddings_and_labels["val_3"][1]
        reference_embeddings = embeddings_and_labels["samples"][0]
        reference_labels = embeddings_and_labels["samples"][1]
        reference_embeddings_2 = embeddings_and_labels["samples_2"][0]
        reference_labels_2 = embeddings_and_labels["samples_2"]
        reference_embeddings_3 = embeddings_and_labels["samples_3"][0]
        reference_labels_3 = embeddings_and_labels["samples_3"]

        #print(reference_labels_1)

        # print(reference_embeddings_1.shape)
        # sys.exit()
        '''
        for i in range(7):
            # mask = reference_labels_1 == i+100
            # mask = np.squeeze(mask)
            reference_embeddings[i, :] = reference_embeddings_1[3 * i:3 * i + 3, :].mean(axis=0)
        reference_labels = np.arange(0, 7)
        '''

        #reference_labels = embeddings_and_labels["samples"][1]
        knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings.astype('float32'),
                                                              query_embeddings.astype('float32'), 1, False)
        knn_labels = reference_labels[knn_indices][:, 0]

        accuracy = accuracy_score(knn_labels, query_labels)
        f_1_score = sklearn.metrics.f1_score(query_labels, knn_labels, average='macro')
        precision = sklearn.metrics.precision_score(query_labels, knn_labels, average='macro')
        recall = sklearn.metrics.recall_score(query_labels, knn_labels, average='macro')
        logging.info('accuracy:{}'.format(accuracy))
        logging.info('f_1_score:{}'.format(f_1_score))
        logging.info('precision:{}'.format(precision))
        logging.info('recall:{}'.format(recall))

        accuracies["accuracy"] = accuracy
        # accuracies["f_1_score"] = f_1_score
        # accuracies["precosion"] = precision
        # accuracies["recall"] = recall
        keyname = self.accuracies_keyname("mean_average_precision_at_r")  # accuracy as keyname not working
        accuracies[keyname] = accuracy
        # print(accuracy

        reference_embeddings = np.zeros((7, 128))
        for i in range(7):
            # mask = reference_labels_1 == i+100
            # mask = np.squeeze(mask)
            reference_embeddings[i, :] = reference_embeddings_2[3 * i:3 * i + 3, :].mean(axis=0)
        reference_labels_2 = np.arange(0, 7)


        knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings.astype('float32'),
                                                              query_embeddings_2.astype('float32'), 1, False)
        knn_labels = reference_labels_2[knn_indices][:, 0]

        accuracy = accuracy_score(knn_labels, query_labels_2)
        f_1_score = sklearn.metrics.f1_score(query_labels_2, knn_labels, average='macro')
        precision = sklearn.metrics.precision_score(query_labels_2, knn_labels, average='macro')
        recall = sklearn.metrics.recall_score(query_labels_2, knn_labels, average='macro')
        logging.info('accuracy:{}'.format(accuracy))
        logging.info('f_1_score:{}'.format(f_1_score))
        logging.info('precision:{}'.format(precision))
        logging.info('recall:{}'.format(recall))

        reference_embeddings = np.zeros((7, 128))
        for i in range(7):
            # mask = reference_labels_1 == i+100
            # mask = np.squeeze(mask)
            reference_embeddings[i, :] = reference_embeddings_3[5 * i:5 * i + 5, :].mean(axis=0)
        reference_labels_3 = np.arange(0, 7)



        knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings.astype('float32'),
                                                              query_embeddings_3.astype('float32'), 1, False)
        knn_labels = reference_labels_3[knn_indices][:, 0]

        accuracy = accuracy_score(knn_labels, query_labels_3)
        f_1_score = sklearn.metrics.f1_score(query_labels_3, knn_labels, average='macro')
        precision = sklearn.metrics.precision_score(query_labels_3, knn_labels, average='macro')
        recall = sklearn.metrics.recall_score(query_labels_3, knn_labels, average='macro')
        logging.info('accuracy:{}'.format(accuracy))
        logging.info('f_1_score:{}'.format(f_1_score))
        logging.info('precision:{}'.format(precision))
        logging.info('recall:{}'.format(recall))
        '''
        query_embeddings = embeddings_and_labels["val"][0]
        query_labels = embeddings_and_labels["val"][1]
        reference_embeddings_1 = embeddings_and_labels["samples"][0]
        reference_labels_1 = embeddings_and_labels["samples"][1]
        # print(reference_labels_1)
        reference_embeddings = np.zeros((7, 128))
        # print(reference_embeddings_1.shape)
        # sys.exit()
        
        for i in range(7):
            # mask = reference_labels_1 == i+100
            # mask = np.squeeze(mask)
            reference_embeddings[i, :] = reference_embeddings_1[3 * i:3 * i + 1, :].mean(axis=0)
            query_embeddings = np.concatenate([query_embeddings, reference_embeddings_1[3 * i + 1:3 * i + 3, :]],
                                              axis=0)
            query_labels = np.concatenate([query_labels, reference_labels_1[3 * i + 1:3 * i + 3, :]], axis=0)
        reference_labels = np.arange(0, 12)

        # reference_labels = embeddings_and_labels["samples"][1]
        knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings.astype('float32'),
                                                              query_embeddings.astype('float32'), 1, False)
        knn_labels = reference_labels[knn_indices][:, 0]

        accuracy = accuracy_score(knn_labels, query_labels)
        f_1_score = sklearn.metrics.f1_score(query_labels, knn_labels, average='macro')
        precision = sklearn.metrics.precision_score(query_labels, knn_labels, average='macro')
        recall = sklearn.metrics.recall_score(query_labels, knn_labels, average='macro')
        logging.info('accuracy:{}'.format(accuracy))
        logging.info('f_1_score:{}'.format(f_1_score))
        logging.info('precision:{}'.format(precision))
        logging.info('recall:{}'.format(recall))
        '''


    def do_knn_and_accuracies_aug(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
        #print(embeddings_and_labels)
        print("test")
        train_embeddings = embeddings_and_labels['train'][0]
        train_labels = embeddings_and_labels['train'][1]
        #print(train_labels.shape)
        #print(train_embeddings.shape)
        base_means = []
        base_cov = []
        for key in range(100):
            feature = train_embeddings[np.squeeze(train_labels == key,axis=1),:]
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)
            #print(cov.shape)
        x_aug, y_aug = calibration_augmentation(base_means, base_cov,embeddings_and_labels)
        #print(train_embeddings.shape)
        #print(train_labels.shape)
        reference_embedding = []
        reference_labels = []
        for key in range(20):
            reference_embedding.append(x_aug[np.squeeze(y_aug==key),:].mean(0))
            reference_labels.append(key)
        reference_embeddings = np.stack(reference_embedding, axis=0)
        reference_labels = np.stack(reference_labels,axis=0)
        query_embeddings = embeddings_and_labels["val"][0]
        query_labels = embeddings_and_labels["val"][1]
        #reference_embeddings = #embeddings_and_labels["samples"][0]
        #reference_labels = #embeddings_and_labels["samples"][1]
        #print(reference_embeddings.shape)
        #print(query_embeddings.shape)
        knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings.astype(np.float32), query_embeddings, 1, False)
        knn_labels = reference_labels[knn_indices][:,0]
        accuracy = accuracy_score(knn_labels, query_labels)
        f_1_score = sklearn.metrics.f1_score(knn_labels, query_labels)
        precision = sklearn.metrics.precision_score(knn_labels, query_labels)
        recall = sklearn.metrics.recall_score(knn_labels, query_labels)
        logging.info('accuracy:{}'.format(accuracy))
        logging.info('f_1_score:{}'.format(f_1_score))
        logging.info('precision:{}'.format(precision))
        logging.info('recall:{}'.format(recall))
        #print('accuracy:', accuracy, ' f_1_score: ',f_1_score, ' precision: 'precision, ' recall: ', recall)

        accuracies["accuracy"] = accuracy
        #accuracies["f_1_score"] = f_1_score
        #accuracies["precosion"] = precision
        #accuracies["recall"] = recall
        keyname = self.accuracies_keyname("mean_average_precision_at_r") # accuracy as keyname not working
        accuracies[keyname] = accuracy

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        out = self.net(x)
        #print(out.size())
        return out
class SignalModule(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #self.trunk_signal = torchvision.models.__dict__[cfg.model.model_name](pretrained=cfg.model.pretrained)
        #ViT(image_size=256,patch_size=64, num_classes=21, dim=512,depth=6, heads=16, mlp_dim=21,dropout=0.1,emb_dropout=0.1)

        self.trunk_signal = LeViT_384(num_classes=512, distillation=True,
              pretrained=False, fuse=False)
        self.trunk_fft = LeViT_384(num_classes=512, distillation=True,
              pretrained=False, fuse=False, in_chans=6)


        """PyramidVisionTransformer(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        )
        self.trunk_fft = PyramidVisionTransformer(
            patch_size=4, in_chans=6, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        )"""

        """
        self.trunk_fft = LeViT(
            image_size = 256,
            num_classes = 20,
            stages = 3,             # number of stages
            dim = (256, 384, 512),  # dimensions at each stage
            depth = 4,              # transformer of depth 4 at each stage
            heads = (4, 6, 8),      # heads at each stage
            mlp_mult = 2,
            dropout = 0.1
         )
        """

        #self.trunk_signal.fc = Identity()
        #self.trunk_fft.fc =Identity()
        #ViT(image_size=256,patch_size=64, num_classes=21, dim=512,depth=6, heads=16,channels=6, mlp_dim=21,dropout=0.1,emb_dropout=0.1)
        #self.conv_fusion = nn.Sequential(
        #    nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(1024, eps=1e-3, momentum=0.01),
        #    nn.ReLU()
        #)
        self.MLP_2 = MLP([1024,512])
        #self.MLP = MLP([6,3])

    def forward(self, x):
        #print(x.size())
        batch_size = x.size()[0]
        c_1 = self.trunk_signal(x)
        
        data_fft = torch.rfft(x.permute(0,1,3,2),signal_ndim=1)
        #print(data_fft[:,:,:,:,1])
        #sys.exit()
        real = data_fft[:,:,:,:,0].permute(0,1,3,2)
        imag = data_fft[:,:,:,:,1].permute(0,1,3,2)
        norm = torch.nn.functional.normalize(torch.sqrt(torch.pow(real,2)+torch.pow(imag,2)), dim=-2)
        angle =torch.nn.functional.normalize(torch.atan2(real,imag),dim=-2)
        data_fusion = torch.cat([norm,angle], dim=1)
        #print(data_fusion.size())
        #sys.exit()
        container = torch.zeros([batch_size,6,256,256]).cuda()
        container[:,:,:129,:]=data_fusion[:,:,:,:]
        container[:,:,129:,:]=torch.flip(container,dims=[2])[:,:,:127,:]
        c_2 = self.trunk_fft(container)
        c_2 = self.trunk_signal(x)
        #print(c_1.size(), c2.size())
        fusion = self.MLP_2(torch.cat([c_1, c_2],dim=1))
        
        
        return fusion




# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_datasets(data_dir, cfg, mode="train"):

    common_transforms = []
    train_transforms = []
    test_transforms = []
    #if cfg.transform.transform_resize_match:
    common_transforms.append(transforms.Resize((256, 256)))
    
    if cfg.transform.transform_random_resized_crop:
        train_transforms.append(transforms.RandomResizedCrop(cfg.transform.transform_resize))
    if cfg.transform.transform_random_horizontal_flip:
        train_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    if cfg.transform.transform_random_rotation:
        train_transforms.append(transforms.RandomRotation(cfg.transform.transform_random_rotation_degrees))#, fill=255))
    if cfg.transform.transform_random_shear:
        train_transforms.append(torchvision.transforms.RandomAffine(0,
                                                                    shear=(
                                                                        cfg.transform.transform_random_shear_x1,
                                                                        cfg.transform.transform_random_shear_x2,
                                                                        cfg.transform.transform_random_shear_y1,
                                                                        cfg.transform.transform_random_shear_y2
                                                                        ),
                                                                    fillcolor=255)) 
    if cfg.transform.transform_random_perspective:
        train_transforms.append(transforms.RandomPerspective(distortion_scale=cfg.transform.transform_perspective_scale, 
                                     p=0.5, 
                                     interpolation=3)
                                )
    if cfg.transform.transform_random_affine:
        train_transforms.append(transforms.RandomAffine(degrees=(cfg.transform.transform_degrees_min,
                                                                 cfg.transform.transform_degrees_max),
                                                        translate=(cfg.transform.transform_translate_a,
                                                                   cfg.transform.transform_translate_b),
                                                        fillcolor=255))

    data_transforms = {
            'train': transforms.Compose(common_transforms+train_transforms+[transforms.ToTensor()]),
            'test': transforms.Compose(common_transforms+[transforms.ToTensor()]),
            }

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"),
            data_transforms["train"])





    # for the final model we can join train, validation, validation samples datasets
    print(mode)
    if mode == "final_train":
        #train_dataset = torch.utils.data.ConcatDataset([train_dataset,
        #        val_dataset,
        #        val_samples_dataset])

        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"),
                data_transforms["test"])

        samples_dataset = datasets.ImageFolder(os.path.join(data_dir, "samples"),
                data_transforms["test"])
        test_dataset_2 = datasets.ImageFolder(os.path.join(data_dir, "test_2"),
                data_transforms["test"])

        samples_dataset_2 = datasets.ImageFolder(os.path.join(data_dir, "samples_2"),
                data_transforms["test"])
        test_dataset_3 = datasets.ImageFolder(os.path.join(data_dir, "test_3"),
                data_transforms["test"])

        samples_dataset_3 = datasets.ImageFolder(os.path.join(data_dir, "samples_3"),
                data_transforms["test"])
        return train_dataset, test_dataset, samples_dataset, test_dataset_2, samples_dataset_2, test_dataset_3, samples_dataset_3
    else:
        if mode == "train":
            val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"),
                    data_transforms["test"])

            val_samples_dataset = datasets.ImageFolder(os.path.join(data_dir, "val_samples"),
                    data_transforms["test"])
            return train_dataset, val_dataset, val_samples_dataset

        if mode == "test":
            return train_dataset, test_dataset, samples_dataset


@hydra.main(config_path="config/config.yaml")
def train_app(cfg):
    print(cfg.pretty())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #trunk = PyramidVisionTransformer(
    #    patch_size=4, embed_dims=[128, 256, 512, 768], num_heads=[2, 4, 8, 12], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
    #    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 10, 60, 3], sr_ratios=[8, 4, 2, 1],
    #     )
    #trunk = PyramidVisionTransformer(
    #    patch_size=32, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
    #    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
    #    )
    #trunk = rest_small(pretrained=False)#
    #trunk = build_model() #LeViT_384(num_classes=512, distillation=True,pretrained=False, fuse=False)
    trunk = LeViT_384(num_classes=512, distillation=True,
              pretrained=False, fuse=False)
    #device=torch.device("cpu")
    # Set trunk model and replace the softmax layer with an identity function
    #trunk = torchvision.models.__dict__[cfg.model.model_name](pretrained=cfg.model.pretrained)
    #trunk = SignalModule(cfg)
    #trunk=ViT(image_size=256,patch_size=64, num_classes=21, dim=512,depth=6, heads=16, mlp_dim=21,dropout=0.1,e
    
    #trunk= torchvision.models.__dict__[cfg.model.model_name](pretrained=cfg.model.pretrained)
    #trunk = models.resnet18(pretrained=False)
    #trunk = models.alexnet(pretrained=True)
    #trunk = models.resnet50(pretrained=True)
    #trunk = models.resnet152(pretrained=True)
    #trunk = models.wide_resnet50_2(pretrained=True)
    #trunk = EfficientNet.from_pretrained('efficientnet-b2')
    #trunk = ViT('B_16_imagenet1k', pretrained=True)
    trunk_output_size = 512
    #trunk.head = Identity()
    #trunk.head_2 = Identity()
    trunk = torch.nn.DataParallel(trunk.to(device))
    
    embedder = torch.nn.DataParallel(MLP([trunk_output_size, cfg.embedder.size]).to(device))
    classifier = torch.nn.DataParallel(MLP([cfg.embedder.size, 23])).to(device) #23 levitpmbfa toyota 24 swin

    # Set optimizers
    if cfg.optimizer.name == "sdg":
        trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "rmsprop":
        trunk_optimizer = torch.optim.RMSprop(trunk.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        embedder_optimizer = torch.optim.RMSprop(embedder.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        classifier_optimizer = torch.optim.RMSprop(classifier.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == 'adam': 
        trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)
        embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)


    # Set the datasets
    data_dir = os.environ["DATASET_FOLDER"]+"/"+cfg.dataset.data_dir
    print("Data dir: "+data_dir)

    train_dataset, val_dataset, val_samples_dataset,  val_dataset_2, val_samples_dataset_2,val_dataset_3, val_samples_dataset_3= get_datasets(data_dir, cfg, mode=cfg.mode.type)
    print("Trainset: ",len(train_dataset), "Testset: ",len(val_dataset), "Samplesset: ",len(val_samples_dataset))

    # Set the loss function
    if cfg.embedder_loss.name == "margin_loss":
        loss = losses.MarginLoss(margin=cfg.embedder_loss.margin,nu=cfg.embedder_loss.nu,beta=cfg.embedder_loss.beta)
    #if cfg.embedder_loss.name == "triplet_margin":
    loss = losses.TripletMarginLoss(margin=cfg.embedder_loss.margin)
    #loss_angular = losses.AngularLoss(alpha=40)
    if cfg.embedder_loss.name == "multi_similarity":
        loss = losses.MultiSimilarityLoss(alpha=cfg.embedder_loss.alpha, beta=cfg.embedder_loss.beta, base=cfg.embedder_loss.base)
    #if cfg.embedder_loss.name == "proxyanchor":
    #loss = Loss() #losses.ProxyAnchorLoss(num_classes = 22, embedding_size = cfg.embedder.size).cuda()
    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function

    if cfg.miner.name == "triplet_margin":
        #miner = miners.TripletMarginMiner(margin=0.2)
        miner = miners.TripletMarginMiner(margin=cfg.miner.margin)
    if cfg.miner.name == "multi_similarity":
        miner = miners.MultiSimilarityMiner(epsilon=cfg.miner.epsilon)
        #miner = miners.MultiSimilarityMiner(epsilon=0.05)

    batch_size = cfg.trainer.batch_size
    num_epochs = cfg.trainer.num_epochs
    iterations_per_epoch = cfg.trainer.iterations_per_epoch
    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))


    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
    loss_funcs = {"metric_loss": loss ,"classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": cfg.loss.metric_loss, "classifier_loss": cfg.loss.classifier_loss}


    schedulers = {
            #"metric_loss_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma),
            "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, 10, gamma=cfg.scheduler.gamma),
            "classifier_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, 10, gamma=cfg.scheduler.gamma),
            "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, 10, gamma=cfg.scheduler.gamma),
            } # cfg.scheduler.step_size

    experiment_name = "model_c2_cat_levit_%s_model_%s_cl_%s_ml_%s_miner_%s_mix_ml_%02.2f_mix_cl_%02.2f_resize_%d_emb_size_%d_class_size_%d_opt_%s_lr_%02.2f_m_%02.2f_wd_%02.2f"%(cfg.dataset.name,
                                                                                                  cfg.model.model_name, 
                                                                                                  "cross_entropy", 
                                                                                                  cfg.embedder_loss.name, 
                                                                                                  cfg.miner.name, 
                                                                                                  cfg.loss.metric_loss, 
                                                                                                  cfg.loss.classifier_loss,
                                                                                                  cfg.transform.transform_resize,
                                                                                                  cfg.embedder.size,
                                                                                                  cfg.embedder.class_out_size,
                                                                                                  cfg.optimizer.name,
                                                                                                  cfg.optimizer.lr,
                                                                                                  cfg.optimizer.momentum,
                                                                                                  cfg.optimizer.weight_decay)
    experiment_name = 'levittoyota_rep_twostage__proPMBFA_re_nturgbd_dataset_100_20_noise'
    record_keeper, _, _ = logging_presets.get_record_keeper("logs_c_2_cat/%s"%(experiment_name), "tensorboard_c_2_cat/%s"%(experiment_name))
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"samples": val_samples_dataset, "val": val_dataset,"samples_2": val_samples_dataset_2, "val_2": val_dataset_2,"samples_3": val_samples_dataset_3, "val_3": val_dataset_3}
    model_folder = "example_saved_models_c_2_cat/%s/"%(experiment_name)

    # Create the tester
    tester = OneShotTester(
            end_of_testing_hook=hooks.end_of_testing_hook,
            #size_of_tsne=20
            )
    #tester.embedding_filename=data_dir+"/embeddings_pretrained_triplet_loss_multi_similarity_miner.pkl"
    tester.embedding_filename=data_dir+"/"+experiment_name+".pkl"
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
    trainer = trainers.TrainWithClassifier(models,
            optimizers,
            batch_size,
            loss_funcs,
            mining_funcs,
            train_dataset,
            sampler=sampler,
            lr_schedulers=schedulers,
            dataloader_num_workers = cfg.trainer.batch_size,
            loss_weights=loss_weights,
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook
            )

    trainer.train(num_epochs=num_epochs)

    tester = OneShotTester()

if __name__ == "__main__":
    train_app()
