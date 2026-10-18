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


class SSRE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # self._network = IncrementalNet(args['convnet_type'], False,args=args)
        self._network = IncrementalNet(args,False)
        self._protos = []



    def after_task(self):
        self._known_classes = (self._total_classes)
        self._old_network = (self._network.copy().freeze())
        if hasattr(self._old_network,"module"): 
            self.old_network_module_ptr = (self._old_network.module)
        else:
            self.old_network_module_ptr = (self._old_network)
        if self.args.get("save_checkpoints",True):
            self.save_checkpoint("{}_{}_{}".format(self.args["model_name"],self.args["init_cls"],self.args["increment"]))

    def incremental_train(self,data_manager):
        self.data_manager = data_manager
        next_task = (self._cur_task + 1)
        if next_task == 0:
            expected_known_classes = 0
        else:
            expected_known_classes = (data_manager.get_accumulate_tasksize(next_task - 1))
        if (self._known_classes != expected_known_classes):
            raise RuntimeError(
                "SSRE task state is inconsistent. "
                f"About to start task "
                f"{next_task}, "
                f"but _known_classes="
                f"{self._known_classes}, "
                f"expected "
                f"{expected_known_classes}."
            )

        self._cur_task = (next_task)
        self._total_classes = (self._known_classes + data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = (self._network)
        logging.info("Model Expansion!")

        self._network_expansion()

        logging.info("Learning on {}-{}".format(self._known_classes,self._total_classes))
        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(count_parameters(self._network,True)))

        train_dataset = (
            data_manager.get_dataset(
                np.arange(
                    self._known_classes,
                    self._total_classes,
                ),
                source="train",
                mode="train",
                appendent=(
                    self._get_memory()
                ),
            )
        )


        self.train_loader = (build_train_loader(dataset=train_dataset,
                dataset_name=(self.args["dataset"]),
                batch_size=(self.args["batch_size"]),
                num_workers=(self.args["num_workers"]),
                seed=(int(self.args["seed"]) + self._cur_task)))

        test_dataset = (data_manager.get_dataset(np.arange(0,self._total_classes),source="test",mode="test"))

        self.test_loader = (build_eval_loader(dataset=test_dataset,
                batch_size=(self.args["batch_size"]),
                num_workers=(self.args["num_workers"])))


        if len(self._multiple_gpus) > 1:
            self._network = (nn.DataParallel(self._network,self._multiple_gpus))

        self._train(self.train_loader,self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = (self._network.module)

        logging.info("Model Compression!")

        self._network_compression()

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
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters(
            )), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()
            
        
    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = (build_eval_loader(dataset=idx_dataset,batch_size=(self.args["batch_size"]),num_workers=(self.args["num_workers"])))
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)

            if (len(self._protos) != self._total_classes):
                raise RuntimeError(
                    "SSRE prototype count "
                    "mismatch: "
                    f"{len(self._protos)} != "
                    f"{self._total_classes}"
                )

    def train(self):
        if self._cur_task > 0:
            self._network.eval()
            return
        self._network.train()
    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            losses_clf, losses_fkd, losses_proto = 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device,non_blocking=True)
                targets = targets.to(self._device,dtype=torch.long,non_blocking=True)
                logits, loss_clf, loss_fkd, loss_proto = self._compute_ssre_loss(inputs,targets)
                #Diagnostic loss
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

    def _compute_ssre_loss(self,inputs,targets):
        outputs = self._network(inputs)
        logits = outputs["logits"]
        features = outputs["features"]

        # =========================
        # Initial task
        # =========================
        if (
            self._known_classes == 0):

            loss_clf = (
                F.cross_entropy(
                    logits
                    / self.args["temp"],
                    targets,
                )
            )

            zero = (
                loss_clf.new_zeros(())
            )

            return (
                logits,
                loss_clf,
                zero,
                zero,
            )


        # =========================
        # Incremental task
        # =========================
        if not hasattr(
            self,
            "old_network_module_ptr",
        ):

            raise RuntimeError(
                "SSRE incremental task "
                "requires frozen old network."
            )


        with torch.no_grad():

            features_old = (
                self.old_network_module_ptr
                .extract_vector(
                    inputs
                )
            )


        protos = torch.as_tensor(
            np.asarray(
                self._protos
            ),
            dtype=features.dtype,
            device=self._device,
        )


        with torch.no_grad():

            weights = (
                F.normalize(
                    features,
                    p=2,
                    dim=1,
                    eps=1e-12,
                )
                @
                F.normalize(
                    protos,
                    p=2,
                    dim=1,
                    eps=1e-12,
                ).T
            )

            mask = torch.max(
                weights,
                dim=1,
            )[0]


        loss_clf = (
            F.cross_entropy(
                logits
                / self.args["temp"],
                targets,
                reduction="none",
            )
        )

        loss_clf = torch.mean(
            loss_clf
            * (1 - mask)
        )


        loss_fkd = torch.norm(
            features
            - features_old,
            p=2,
            dim=1,
        )

        loss_fkd = (
            self.args["lambda_fkd"]
            * torch.sum(
                loss_fkd
                * mask
            )
        )


        index = np.random.choice(
            range(
                self._known_classes
            ),
            size=(
                self.args[
                    "batch_size"
                ]
            ),
            replace=True,
        )


        proto_features = (
            np.asarray(
                self._protos
            )[index]
        )

        proto_targets = index


        proto_features = (
            torch.from_numpy(
                proto_features
            )
            .float()
            .to(
                self._device,
                non_blocking=True,
            )
        )


        proto_targets = (
            torch.from_numpy(
                proto_targets
            )
            .long()
            .to(
                self._device,
                non_blocking=True,
            )
        )


        proto_logits = (
            self._network_module_ptr
            .fc(
                proto_features
            )["logits"]
        )


        loss_proto = (
            self.args[
                "lambda_proto"
            ]
            * F.cross_entropy(
                proto_logits
                / self.args["temp"],
                proto_targets,
            )
        )


        return (
            logits,
            loss_clf,
            loss_fkd,
            loss_proto,
        )
        
    
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        elif hasattr(self, '_protos'):
            # y_pred, y_true = self._eval_nme(self.test_loader, self._protos/np.linalg.norm(self._protos,axis=1)[:,None])
            protos = np.asarray(self._protos,dtype=np.float32)
            proto_norms = np.linalg.norm(protos,axis=1,keepdims=True)
            proto_norms = np.clip(proto_norms,1e-12,None)
            normalized_protos = (protos / proto_norms)
            y_pred, y_true = self._eval_nme(self.test_loader,normalized_protos)
            nme_accy = self._evaluate(y_pred, y_true)            
        else:
            nme_accy = None

        return cnn_accy, nme_accy
    
    def _network_expansion(self):
        # Initial task:
        # seluruh backbone + attention
        # boleh dilatih.
        if self._cur_task == 0:

            for parameter in (
                self._network
                .convnet
                .parameters()
            ):
                parameter.requires_grad = True

        # Incremental tasks:
        # hanya parallel adapter yang
        # boleh mengubah representation.
        else:

            for name, parameter in (
                self._network
                .convnet
                .named_parameters()
            ):

                parameter.requires_grad = (
                    "adapter"
                    in name
                )

        self._network.convnet.switch(
            "parallel_adapters"
        )    
        
    def _network_compression(self):
        model_dict = (
            self._network
            .state_dict()
        )

        for key in list(
            model_dict.keys()
        ):

            if (
                ".adapter.weight"
                not in key
            ):
                continue

            conv_key = key.replace(
                ".adapter.weight",
                ".conv.weight",
            )

            adapter_weight = (
                model_dict[key]
            )

            padded_adapter = F.pad(
                adapter_weight,
                [
                    1,
                    1,
                    1,
                    1,
                ],
                "constant",
                0,
            )

            model_dict[
                conv_key
            ] = (
                model_dict[
                    conv_key
                ]
                + padded_adapter
            )

            model_dict[
                key
            ] = torch.zeros_like(
                adapter_weight
            )


        self._network.load_state_dict(
            model_dict
        )

        self._network.convnet.switch(
            "normal"
        )