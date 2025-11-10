from utils.tools import *
import itertools
from scipy.linalg import hadamard
from network import *
import pdb
import os

import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--hash_dim", type=int, default=32)
parser.add_argument("--noise_rate", type=float, default=1.0)
parser.add_argument("--dataset", type=str, default="flickr")
parser.add_argument("--num_gradual", type=int, default=100)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--margin", type=float, default=0.2)
parser.add_argument("--shift", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=0.7)

args = parser.parse_args()


bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
num_gradual = args.num_gradual
k=args.k
margin = args.margin
shift = args.shift
alpha = args.alpha
beta = args.beta

if dataset == "flickr":
    train_size = 10000
elif dataset == "ms-coco":
    train_size = 10000
elif dataset == "nuswide21":
    train_size = 10500
elif dataset == "iapr":
    train_size = 10000
n_class = 0
tag_len = 0

torch.multiprocessing.set_sharing_strategy("file_system")

def get_config():
    config = {
        "optimizer": {
            "type": optim.Adam,
            "optim_params": {"lr": 1e-4, "weight_decay": 10**-5},
        },
        "txt_optimizer": {
            "type": optim.Adam,
            "optim_params": {"lr": 1e-4, "weight_decay": 10**-5},
        },
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "dataset": dataset,
        "epoch": 50,
        "device": torch.device("cuda:"+args.gpus),
        "bit_len": bit_len,
        "noise_type": "symmetric",
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "tag_len": tag_len,
        "train_size": train_size,
        "margin" :margin,
        "shift" :shift,
        "alpha" :alpha,
        "beta" :beta,
        "k" :k
    }
    return config


class MultiLabelLoss(nn.Module):
    def __init__(
        self,
        margin=0.2,
        lambda_contrast=0.7,
        lambda_quant=0.3,
        lambda_class=1.0,
        tau=1.0,
        shift=1.0,
    ):
        super(MultiLabelLoss, self).__init__()
        self.margin = margin
        self.lambda_contrast = lambda_contrast
        self.lambda_quant = lambda_quant
        self.lambda_class=lambda_class
        self.tau = tau
        self.shift = shift

    def compute_label_similarity(self, label):
        intersection = torch.matmul(label, label.T)
        union = (
            torch.sum(label, dim=1, keepdim=True)
            + torch.sum(label, dim=1, keepdim=True).T
            - intersection
        )
        return intersection / (union + 1e-8)

    def contrast_loss(self, u, v, label):

        T = self.compute_label_similarity(label)

        S = torch.matmul(u, v.T)
        diag_s = torch.diag(S).unsqueeze(1)

        mask_te = (S >= (diag_s - self.margin)).float().detach()
        cost_te = torch.where(mask_te.bool(), S, S - self.shift)
        diag_te = torch.diag(cost_te).clamp(min=0)
        cost_te = cost_te - torch.diag(torch.diag(cost_te)) + torch.diag(diag_te)

        mask_im = (S >= (diag_s.T - self.margin)).float().detach()
        cost_im = torch.where(mask_im.bool(), S, S - self.shift)
        diag_im = torch.diag(cost_im).clamp(min=0)
        cost_im = cost_im - torch.diag(torch.diag(cost_im)) + torch.diag(diag_im)

        loss_r = (
            self.tau * ((cost_te / self.tau * (1 - T)).exp()) +
            self.tau * ((cost_im / self.tau * (1 - T)).exp())
        ).mean()

        positive_mask = ((T- torch.eye(T.shape[0],device=T.device,dtype=T.dtype)) > 0).float()
        separation_term = torch.mean(positive_mask * (self.shift - S).exp())

        return loss_r + separation_term - torch.diag(S).mean()
    
    def quant_loss(self, u, v, label=None, label_confidence=None):
        loss_u = torch.norm(u - u.sign(), p=1) / u.numel()
        loss_v = torch.norm(v - v.sign(), p=1) / v.numel()
        return loss_u + loss_v

    def classification_loss(self,p_u, p_v, label,weights):

        loss_cls_u = F.binary_cross_entropy(p_u, label, reduction='none') 
        loss_cls_v = F.binary_cross_entropy(p_v, label, reduction='none') 

        loss_cls_u = loss_cls_u.mean(dim=1)  # [batch_size]
        loss_cls_v = loss_cls_v.mean(dim=1)  # [batch_size]

        classification_loss = (loss_cls_u + loss_cls_v) / 2.0  # [batch_size]

        if weights is not None:
            loss = (classification_loss * weights).mean()
        else:
            loss = (classification_loss).mean()

        return loss
    
    def forward(self, u, v, label,p_u,p_v,clean_sample=None):
        loss_contrast = self.contrast_loss(u, v, label)
        loss_quant = self.quant_loss(u, v)
        loss_class=self.classification_loss(p_u,p_v,label,clean_sample)
        return self.lambda_contrast * loss_contrast + self.lambda_quant * loss_quant +loss_class*self.lambda_class,loss_class,loss_contrast

def select_clean_sample(data_loader,img_net,txt_net,k,device,gamma=0.5):
    u_features=[]
    v_features=[]
    all_index=[]
    all_labels=[]

    all_t_label=[]
    img_net.eval()
    txt_net.eval()
    for imgs,txts,t_label,label,index in data_loader:
        imgs = imgs.float().to(device)
        txts = txts.float().to(device)
        label = label.to(device)
        u = img_net(imgs)
        v = txt_net(txts)
        u_features.append(u)
        v_features.append(v)
        all_index.append(index)
        all_labels.append(label)
        all_t_label.append(t_label)
    sorted_indices = torch.argsort(torch.cat(all_index))
    u_features = torch.cat(u_features)[sorted_indices]
    v_features = torch.cat(v_features)[sorted_indices]
    all_labels = torch.cat(all_labels)[sorted_indices]
    all_t_label = torch.cat(all_t_label)[sorted_indices]

    u_norm = torch.nn.functional.normalize(u_features, dim=1)
    u_sim_matrix = torch.mm(u_norm, u_norm.t()).fill_diagonal_(0)

    v_norm = torch.nn.functional.normalize(v_features, dim=1)
    v_sim_matrix = torch.mm(v_norm, v_norm.t()).fill_diagonal_(0)

    sim = (u_sim_matrix + v_sim_matrix) / 2

    top_sim, nearest_indices = torch.topk(sim, k=k, dim=1)
    top_sim_norm = top_sim / (top_sim.sum(dim=1, keepdim=True) + 1e-8)
    nearest_labels = all_labels[nearest_indices] * top_sim_norm.unsqueeze(-1)
    new_label = nearest_labels.sum(dim=1)

    label_cosine_sim = F.cosine_similarity(all_labels, new_label, dim=1)

    clean_weight=gamma+(1-gamma)*(label_cosine_sim)      
    
    return clean_weight.detach()

def train(config, bit, seed,aa):

    device = config["device"]
    # alpha = config["alpha"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = (
        get_data(config)
    )
    config["num_train"] = num_train
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3, num_classes=n_class).to(device)
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2, num_classes=n_class).to(
        device
    )
    W_u = torch.Tensor(bit_len, n_class)
    W_u = torch.nn.init.orthogonal_(W_u, gain=1)
    W_u = W_u.clone().detach().requires_grad_(True).to(device)
    W_u = torch.nn.Parameter(W_u)
    net.register_parameter("W_u", W_u)  # regist W_v into the image net

    W_v = torch.Tensor(bit_len, n_class)
    W_v = torch.nn.init.orthogonal_(W_v, gain=1)
    W_v = W_v.clone().detach().requires_grad_(True).to(device)
    W_v = torch.nn.Parameter(W_v)
    txt_net.register_parameter("W_v", W_v)  # regist W into the txt net
    get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
    params_dnet = get_grad_params(net)
    optimizer = config["optimizer"]["type"](
        params_dnet, **(config["optimizer"]["optim_params"])
    )
    txt_optimizer = config["txt_optimizer"]["type"](
        txt_net.parameters(), **(config["txt_optimizer"]["optim_params"])
    )
    
    criterion = MultiLabelLoss(shift=config["shift"],margin=config["margin"], lambda_contrast=config["alpha"],lambda_quant=config["beta"]).to(device)

    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    bestt2i = 0
    besti2t = 0

    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./other", exist_ok=True)
    os.makedirs("./PR", exist_ok=True)



    with open(
        "./logs/data_{}_seed_{}_noiseRate_{}_bit_{}.txt".format(
            config["dataset"],
            seed,
            config["noise_rate"],
            bit,
        ),
        "w",
    ) as f:
        for epoch in range(config["epoch"]):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print(
                "%s[%2d/%2d][%s] bit:%d, dataset:%s, training...."
                % (
                    config["info"],
                    epoch + 1,
                    config["epoch"],
                    current_time,
                    bit,
                    config["dataset"],
                ),
                end="",
            )
            net.eval()
            txt_net.eval()

            train_loss = 0
            if (epoch + 1) % 1 == 0:
                print("calculating test binary code......")
                img_tst_binary, img_tst_label = compute_img_result(
                    test_loader, net, device=device
                )
                print("calculating dataset binary code.......")
                img_trn_binary, img_trn_label = compute_img_result(
                    dataset_loader, net, device=device
                )
                txt_tst_binary, txt_tst_label = compute_tag_result(
                    test_loader, txt_net, device=device
                )
                txt_trn_binary, txt_trn_label = compute_tag_result(
                    dataset_loader, txt_net, device=device
                )
                print("calculating map.......")
                t2i_mAP = calc_map_k(
                    img_trn_binary.numpy(),
                    txt_tst_binary.numpy(),
                    img_trn_label.numpy(),
                    txt_tst_label.numpy(),
                    device=device,
                )
                                
                t2i_r, t2i_p = pr_curve(
                    img_trn_binary.numpy(),
                    txt_tst_binary.numpy(),
                    img_trn_label.numpy(),
                    txt_tst_label.numpy(),
                    device=device,
                )
                i2t_mAP = calc_map_k(
                    txt_trn_binary.numpy(),
                    img_tst_binary.numpy(),
                    txt_trn_label.numpy(),
                    img_tst_label.numpy(),
                    device=device,
                )
                i2t_r, i2t_p = pr_curve(
                    txt_trn_binary.numpy(),
                    img_tst_binary.numpy(),
                    txt_trn_label.numpy(),
                    img_tst_label.numpy(),
                    device=device,
                )
                if t2i_mAP + i2t_mAP > bestt2i + besti2t:
                    bestt2i = t2i_mAP
                    besti2t = i2t_mAP
                    bestt2i_r = t2i_r
                    bestt2i_p = t2i_p
                    besti2t_r = i2t_r
                    besti2t_p = i2t_p
                    data_to_save = {
                        "bestt2i_r": bestt2i_r,
                        "bestt2i_p": bestt2i_p,
                        "besti2t_r": besti2t_r,
                        "besti2t_p": besti2t_p,
                    }
                    with open(
                        "./PR/data_{}_seed_{}_noiseRate_{}_bit_{}_best_PR.pkl".format(
                            config["dataset"],
                            seed,
                            config["noise_rate"],
                            bit,
                        ),
                        "wb",
                    ) as f1:
                        pickle.dump(data_to_save, f1)
                t2i_mAP_list.append(t2i_mAP.item())
                i2t_mAP_list.append(i2t_mAP.item())
                epoch_list.append(epoch)
                print(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f \n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
                f.writelines(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f\n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
            net.train()
            txt_net.train()
            cr_loss=0
            cl_loss=0
            if epoch>=5:
                sample_clean_weight = select_clean_sample(train_loader, net, txt_net, config["k"],device)

            for i, (image, tag, tlabel, label, ind) in enumerate(train_loader):
                image = image.float().to(device)
                tag = tag.float().to(device)
                label = label.float().to(device)
                optimizer.zero_grad()
                txt_optimizer.zero_grad()
                u = net(image)
                v = txt_net(tag)

                p_u = torch.sigmoid(torch.matmul(u, W_u))  # [batch_size, num_class]
                p_v = torch.sigmoid(torch.matmul(v, W_v))  # [batch_size, num_class]

                if epoch>=5:
                    clean_sample = sample_clean_weight[ind]

                    loss,loss_cr,loss_class = criterion(
                        u, v, label,p_u,p_v,clean_sample
                    )
                else:
                    loss,loss_cr,loss_class = criterion(
                        u, v, label,p_u,p_v
                    )
                loss.backward()
                optimizer.step()
                txt_optimizer.step()

                cr_loss+=loss_cr.item()
                cl_loss+=loss_class.item()

            print("cross_loss:{}, class_loss:{}".format(cr_loss / len(train_loader),cl_loss/len(train_loader)))


        f.writelines(
            f"best result : bit:{bit}, dataset:{config['dataset']}, noise_rate:{config['noise_rate']:.1f}, t2i_mAP:{bestt2i:.3f}, i2t_mAP:{besti2t:.3f}, average:{(besti2t + bestt2i) / 2.0 * 100.0:.1f}\n"
        )


def test(config, bit, model_path="./checkpoint/best_model.pth"):
    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3).to(device)
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2).to(device)
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad=True).to(device)
    W = torch.nn.Parameter(W)
    net.register_parameter("W", W)
    # Load the saved models
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["net_state_dict"])
    txt_net.load_state_dict(checkpoint["txt_net_state_dict"])
    net.eval()
    txt_net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(
        dataset_loader, net, device=device
    )
    txt_tst_binary, txt_tst_label = compute_tag_result(
        test_loader, txt_net, device=device
    )
    txt_trn_binary, txt_trn_label = compute_tag_result(
        dataset_loader, txt_net, device=device
    )
    print("calculating map.......")
    t2i_mAP = calc_map_k(
        img_trn_binary.numpy(),
        txt_tst_binary.numpy(),
        img_trn_label.numpy(),
        txt_tst_label.numpy(),
        device=device,
    )
    i2t_mAP = calc_map_k(
        txt_trn_binary.numpy(),
        img_tst_binary.numpy(),
        txt_trn_label.numpy(),
        img_tst_label.numpy(),
        device=device,
    )
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))


if __name__ == "__main__":
    data_name_list = ["ms-coco","flickr","iapr","nuswide21"]
    bit_list = [16,32,64,128]
    noise_rate_list = [0.2,0.5,0.8]
    for rand_num in [123]:
        for data_name in data_name_list:
            for rate in noise_rate_list:
                for bit in bit_list:
                    for a in [0]:
                        setup_seed(rand_num)
                        bit_len = bit
                        noise_rate = rate
                        dataset = data_name
                        if dataset == "nuswide21":
                            n_class = 21
                            tag_len = 1000

                            k=20
                            margin = 0.2
                            shift = 1.0

                        elif dataset == "flickr":
                            n_class = 24
                            tag_len = 1386

                            k=20
                            margin = 0.15
                            shift = 0.8

                        elif dataset == "ms-coco":
                            n_class = 80
                            tag_len = 300

                            k=20
                            margin = 0.3
                            shift = 1.1

                        elif dataset == "iapr":
                            n_class = 255
                            tag_len = 2912
                                                       
                            k=20
                            margin = 0.3
                            shift = 1.2
                            
                        config = get_config()
                        print(config)
                        train(config, bit, rand_num,a)
                        # test(config, bit)
