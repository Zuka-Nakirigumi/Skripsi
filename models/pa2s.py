import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, FOSTERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

from thesis.data.loaders import (
    build_train_loader,
    build_eval_loader,
)

EPSILON = 1e-8


class PASS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self._protos = []
        self._radius = 0
        self._radiuses = []

    ### Rotation helper
    def _expand_rotations(self,inputs,targets):
        """
        Membentuk empat rotasi:
        0°, 90°, 180°, 270°.

        Input:
            inputs  : [B, C, H, W]
            targets : [B]

        Output:
            inputs  : [4B, C, H, W]
            targets : [4B]
        """

        if inputs.ndim != 4:
            raise ValueError(
                "PASS expects input shape "
                "[B, C, H, W]."
            )

        batch_size, channels, height, width = (inputs.shape)
        targets = targets.long()

        rotated_inputs = torch.stack(
            [
                torch.rot90(
                    inputs,
                    k,
                    dims=(2, 3),
                )
                for k in range(4)
            ],
            dim=1,
        )

        rotated_inputs = (rotated_inputs.reshape(batch_size * 4,channels,height,width))
        rotated_targets = torch.stack([targets * 4 + k for k in range(4)],dim=1,).reshape(-1)

        return (rotated_inputs,rotated_targets)
    
    def after_task(self):
        # ==========================================
        # 1. Commit jumlah kelas yang sudah dipelajari
        # ==========================================
        self._known_classes = (self._total_classes)

        # ==========================================
        # 2. Buat frozen teacher untuk task berikutnya
        # ==========================================
        self._old_network = (self._network.copy().freeze())
        # Pointer yang dipakai oleh
        # feature knowledge distillation.
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = (self._old_network.module)
        else:
            self.old_network_module_ptr = (self._old_network)

        # ==========================================
        # 3. Checkpoint bersifat opsional
        # ==========================================
        if self.args.get("save_checkpoints",True,):
            checkpoint_prefix = (self.args.get("checkpoint_prefix","{}_{}_{}".format(self.args["model_name"],self.args["init_cls"],self.args["increment"],),))
            self.save_checkpoint(checkpoint_prefix)

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
    
        #Consistency guard
        next_task = (self._cur_task + 1)
        if next_task == 0:
            expected_known_classes = 0
        else:
            expected_known_classes = (data_manager.get_accumulate_tasksize(next_task - 1))
        if (self._known_classes != expected_known_classes):
            raise RuntimeError(
                "PASS task state is inconsistent. "
                f"About to start task {next_task}, "
                f"but _known_classes="
                f"{self._known_classes}, "
                f"expected "
                f"{expected_known_classes}. "
                "Did you forget to call "
                "after_task() after the "
                "previous task?"
            )

        self._cur_task = next_task
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes*4)
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = (build_train_loader(dataset=train_dataset,dataset_name=(self.args["dataset"]),
                                                batch_size=(self.args["batch_size"]),
                                                num_workers=(self.args["num_workers"]),
                                                seed=(int(self.args["seed"])+ self._cur_task)))

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = (build_eval_loader(dataset=test_dataset,
                                            batch_size=(self.args["batch_size"]),
                                            num_workers=(self.args["num_workers"])))
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        
        resume = False
        if self._cur_task in []:
            self._network.load_state_dict(torch.load("{}_{}_{}_{}.pkl".format(self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            resume = True
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if not resume:
            self._epoch_num = self.args["epochs"]
            optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()
            
        
    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                ### Prototype extraction doesnt use weighted random sampler
                idx_loader = (build_eval_loader(dataset=idx_dataset,
                                                batch_size=(self.args["batch_size"]),
                                                num_workers=(self.args["num_workers"])))
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)
                cov = np.cov(vectors.T)
                self._radiuses.append(np.trace(cov)/vectors.shape[1]) 
            self._radius = np.sqrt(np.mean(self._radiuses))
            if (len(self._protos) != self._total_classes):
                raise RuntimeError(
                    "Prototype count mismatch: "
                    f"got {len(self._protos)}, "
                    f"expected "
                    f"{self._total_classes}."
                )
            if (len(self._radiuses) != self._total_classes):
                raise RuntimeError(
                    "Radius count mismatch: "
                    f"got {len(self._radiuses)}, "
                    f"expected "
                    f"{self._total_classes}."
                )

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            losses_clf, losses_fkd, losses_proto = 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                ### Making PASS input dynamic
                batch_size, channels, height, width = inputs.shape
                inputs = inputs.to(self._device,non_blocking=True)
                targets = targets.to(self._device,dtype=torch.long,non_blocking=True)
                inputs, targets = (self._expand_rotations(inputs,targets))
                # inputs = inputs.view(-1,channels,height,width)
                logits, loss_clf, loss_fkd, loss_proto = self._compute_pass_loss(inputs,targets)
                # Diagnostic loss
                self._last_loss_clf = float(loss_clf.detach().cpu())
                self._last_loss_fkd = float(loss_fkd.detach().cpu())
                self._last_loss_proto = float(loss_proto.detach().cpu())
                loss = loss_clf + loss_fkd + loss_proto
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _compute_pass_loss(self,inputs,targets):
        outputs = self._network(
            inputs
        )

        logits = outputs["logits"]
        loss_clf = F.cross_entropy(
            logits / self.args["temp"],
            targets,
        )

        # =========================
        # Initial task
        # =========================
        if self._cur_task == 0:
            zero = loss_clf.new_zeros(())
            return (logits,loss_clf,zero,zero)

        # =========================
        # Incremental tasks
        # =========================

        # Gunakan feature dari forward
        # yang sudah dilakukan di atas.
        features = outputs["features"]  

        # Old network frozen:
        # tidak membutuhkan graph gradient.
        with torch.no_grad():
            features_old = (self.old_network_module_ptr.extract_vector(inputs))

        loss_fkd = (self.args["lambda_fkd"] * torch.dist(features,features_old,2))
        num_new_classes = (self._total_classes - self._known_classes)
        proto_batch_size = (self.args["batch_size"] * int(self._known_classes / num_new_classes))
        index = np.random.choice(range(self._known_classes),size=proto_batch_size,replace=True)
        proto_features = (np.asarray(self._protos)[index])
        proto_targets = (4 * index)
        proto_features = (proto_features + np.random.normal(0,1,proto_features.shape) * self._radius)
        proto_features = (torch.from_numpy(proto_features).float().to(self._device,non_blocking=True))
        proto_targets = (torch.from_numpy(proto_targets).long().to(self._device,non_blocking=True))
        proto_logits = (self._network_module_ptr.fc(proto_features)["logits"])
        loss_proto = (self.args["lambda_proto"] * F.cross_entropy(proto_logits / self.args["temp"],proto_targets))

        return (logits,loss_clf,loss_fkd,loss_proto)
        
    
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:,::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:,::4]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        self._last_canonical_y_pred = (np.asarray(y_pred))
        self._last_canonical_y_true = (np.asarray(y_true))
        self._canonical_eval_name = ("cnn")

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        elif hasattr(self, '_protos'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._protos/np.linalg.norm(self._protos,axis=1)[:,None])
            nme_accy = self._evaluate(y_pred, y_true)            
        else:
            nme_accy = None

        return cnn_accy, nme_accy