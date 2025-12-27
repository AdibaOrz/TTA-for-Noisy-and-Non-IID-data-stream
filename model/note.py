import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from .iabn import convert_iabn, IABN1d, IABN2d
from utils import memory
from utils.loss_functions import HLoss
from utils.logging import to_json
from .ResNet import ResNet18

class NOTE:
    def __init__(self, source_dataset, val_dataset, checkpoint_path, device='cuda',
                 lr=0.1, weight_decay=0.0005, momentum=0.9, batch_size=128, epochs=200,
                 iabn=True, alpha=4, memory_type="PBRS", capacity=64,
                 conf_thresh=0):
        # Data-related
        print(f"Using device: {device}")
        self.device = device
        self.source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.num_classes = source_dataset.num_classes

        self.checkpoint_path = checkpoint_path

        # IABN-related
        self.iabn = iabn
        self.alpha = alpha

        # Train-related
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs

        # Init net
        self.net = ResNet18().to(self.device)
        if self.iabn:
            convert_iabn(self.net, self.alpha)
        num_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(num_feats, self.num_classes).to(self.device)

        # Init train config
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=epochs * len(self.source_dataloader))

        self.class_criterion = torch.nn.CrossEntropyLoss()

        # Manage memory
        if memory_type == "PBRS":
            self.mem = memory.PBRS(capacity=capacity, num_class=self.num_classes)
        elif memory_type == "FIFO":
            self.mem = memory.FIFO(capacity=capacity)
        else:
            raise NotImplementedError
        self.conf_thresh = conf_thresh

        self.json = {}
        self.l2_distance = []
        self.occurred_class = [0 for _ in range(self.num_classes)]

    def train(self):
        self.net.train()

        class_loss_sum = 0
        total_n = 0
        for data in tqdm(self.source_dataloader, total=len(self.source_dataloader)):
            feats, cls, _ = data
            feats = feats.to(self.device)
            cls = cls.to(self.device)

            preds = self.net(feats)

            class_loss = self.class_criterion(preds, cls)
            class_loss_sum += float(class_loss.item() * feats.shape[0])
            total_n += feats.shape[0]

            self.optimizer.zero_grad()
            class_loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        avg_loss = class_loss_sum / total_n
        return avg_loss

    def evaluation(self):
        self.net.eval()

        class_loss_sum = 0
        class_acc_sum = 0
        total_n = 0

        for data in self.val_dataloader:
            feats, cls, _ = data
            feats = feats.to(self.device)
            cls = cls.to(self.device)

            with torch.no_grad():
                preds = self.net(feats)
                class_loss = self.class_criterion(preds, cls)

            class_loss_sum += float(class_loss.item() * feats.shape[0])
            class_acc_sum += (preds.max(1, keepdim=False)[1] == cls).sum()
            total_n += feats.shape[0]

        return class_loss_sum / total_n, class_acc_sum / total_n

    def setup_online(self, target_dataset, save_path,
                     epochs=1, lr=0.0001, weight_decay=0, batch_size=128,
                     use_learned_stats=True, bn_momentum=0.01, update_interval=64,
                     temp_factor=1.0, optimize=True, adapt=True,
                     weighted=False, e_margin=0.4*math.log(10)):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_learned_stats = use_learned_stats
        self.bn_momentum = bn_momentum
        self.update_interval = update_interval
        self.temp_factor = temp_factor
        self.optimize = optimize
        self.adapt = adapt
        self.weighted = weighted
        self.e_margin= e_margin

        self.target_dataset = target_dataset  # Kept as dataset for ease later
        self.save_path = save_path

        # Turn grad off for non-BN layers
        for param in self.net.parameters():
            param.requires_grad = False
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                if use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if isinstance(module, IABN1d) or isinstance(module, IABN2d):
                for param in module.parameters():
                    param.requires_grad = True

        # Init online train config
        self.scheduler = None
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.fifo = memory.FIFO(capacity=self.update_interval)
        if isinstance(self.mem, memory.PBRS):
            self.mem = memory.PBRS(self.mem.capacity, self.num_classes)
        else:
            self.mem = memory.FIFO(self.mem.capacity)

    def train_online(self, sample_num):
        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        N = len(self.target_dataset)
        if sample_num >= N:
            print("EARLY FINISH")
            return FINISHED

        current_sample =  self.target_dataset[sample_num]
        self.fifo.add_instance(current_sample)

        with torch.no_grad():
            self.net.eval()

            if isinstance(self.mem, memory.FIFO):
                self.mem.add_instance(current_sample)
            else:
                f, c, d = current_sample
                f = f.to(self.device)
                c = c.to(self.device)
                d = d.to(self.device)

                logit = self.net(f.unsqueeze(0))
                probs = F.softmax(logit, dim=1)
                confidence, pseudo_cls = probs.max(1, keepdim=False)
                pseudo_cls = pseudo_cls[0]
                confidence = confidence[0]

                if confidence > self.conf_thresh:
                    self.mem.add_instance((f, pseudo_cls, d))

        if self.use_learned_stats:
            self.evaluation_online(sample_num + 1, [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if (sample_num + 1) % self.update_interval != 0:
                if not (sample_num == len(self.target_dataset)-1
                        and self.update_interval > sample_num):
                    return SKIPPED

        if not self.use_learned_stats:
            self.evaluation_online(sample_num + 1, self.fifo.get_memory())

        if not self.adapt:
            print("NO ADAPTATION")
            return TRAINED

        self.net.train()

        if len(self.target_dataset) == 1:
            self.net.eval()

        feats, _, _ = self.mem.get_memory()
        feats = torch.stack(feats)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )
        entropy_loss = HLoss(temp_factor=self.temp_factor, weighted=self.weighted, e_margin=self.e_margin)

        for epoch in range(self.epochs):
            for feats in data_loader:
                feats = feats[0].to(self.device)
                preds = self.net(feats)

                if self.optimize:
                    loss = entropy_loss(preds)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        return TRAINED

    def evaluation_online(self, epoch, current_sample):
        self.net.eval()

        with torch.no_grad():
            feats, cls, do = current_sample

            feats, cls, do = torch.stack(feats), torch.stack(cls), torch.stack(do)
            feats, cls, do = feats.to(self.device), cls.to(self.device), do.to(self.device)

            out = F.softmax(self.net(feats), dim=1)
            conf, y_pred =out.max(1, keepdim=False)

            try:
                true_cls_list = self.json['gt']
                pred_cls_list = self.json['pred']
                accuracy_list = self.json['accuracy']
                f1_macro_list = self.json['f1_macro']
                distance_l2_list = self.json['distance_l2']
                conf_list = self.json['confidence']
            except KeyError:
                true_cls_list = []
                pred_cls_list = []
                accuracy_list = []
                f1_macro_list = []
                distance_l2_list = []
                conf_list = []

            # append values to lists
            true_cls_list += [int(c) for c in cls]
            pred_cls_list += [int(c) for c in y_pred.tolist()]
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) / float(
                len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)
            f1_macro_list.append(f1_score(true_cls_list, pred_cls_list,
                                          average='macro'))
            conf_list += [float(x) for x in conf.tolist()]

            self.occurred_class = [0 for _ in range(self.num_classes)]

            progress_checkpoint = [int(i * (len(self.target_dataset) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_sample[0]), epoch + 1):  # consider a batch input
                if i in progress_checkpoint:
                    print(
                        f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

            # update self.json file
            self.json = {
                'gt': true_cls_list,
                'pred': pred_cls_list,
                'accuracy': accuracy_list,
                'f1_macro': f1_macro_list,
                'distance_l2': distance_l2_list,
                'confidence': conf_list,
            }

    def dump_eval_online_result(self, is_train_offline=False):
        if is_train_offline:
            data_loader = torch.utils.data.DataLoader(self.target_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False)

            for i, data in enumerate(data_loader):
                feats, cls, do = data
                input_data = [list(feats), list(cls), list(do)]
                self.evaluation_online(i * self.batch_size, input_data)

        # logging json files
        json_file = open(os.path.join(self.save_path, 'online_eval.json'), 'w')
        json_subsample = {key: self.json[key] for key in self.json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def save_checkpoint(self, epoch):
        ckpt = {'model': self.net.state_dict(),
                'epoch': epoch,}
        torch.save(ckpt, os.path.join(self.checkpoint_path, "pretrained_checkpoint.pth"))

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.checkpoint_path, "pretrained_checkpoint.pth")
        if os.path.isfile(ckpt_path):
            print(f"Loading model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, weights_only=False)
            self.net.load_state_dict(ckpt['model'])
            return ckpt['epoch']
        return -1