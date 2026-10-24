import logging
import numpy as np
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

from models.base import BaseLearner

from utils.inc_net import (
    CosineIncrementalNet,
)

from utils.toolkit import (
    count_parameters,
    tensor2numpy,
)

from tqdm import tqdm

from thesis.data.loaders import (
    build_train_loader,
    build_eval_loader,
)

class FeCAM(BaseLearner):
    def __init__(self,args):
        super().__init__(args)

        self.args = args

        self._network = (
            CosineIncrementalNet(
                args,
                False,
            )
        )

        self._protos = []

        self._init_protos = []

        self._cov_mat = []

        self._cov_mat_shrink = []

        self._norm_cov_mat = []

        self._diag_mat = []

        self._common_cov = None

    def after_task(self):
        self._known_classes = (self._total_classes)

        if self.args.get("save_checkpoints",True,):
            checkpoint_prefix = (
                self.args.get(
                    "checkpoint_prefix",
                    "{}_{}_{}".format(self.args["model_name"],self.args["init_cls"],self.args["increment"],),))
            self.save_checkpoint(checkpoint_prefix)

    def incremental_train(self,data_manager):
        self.data_manager = (data_manager)
        next_task = (self._cur_task + 1)

        if next_task == 0:
            expected_known_classes = 0

        else:
            expected_known_classes = (data_manager.get_accumulate_tasksize(next_task - 1))

        if (self._known_classes != expected_known_classes):

            raise RuntimeError(
                "FeCAM task state is inconsistent. "
                f"About to start task {next_task}, "
                f"but _known_classes="
                f"{self._known_classes}, "
                f"expected "
                f"{expected_known_classes}."
            )

        self._cur_task = (next_task)

        self._total_classes = (self._known_classes + data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes,self._cur_task)
        self._network_module_ptr = (self._network)

        # Setelah Task 0 feature extractor
        # harus frozen.
        if self._cur_task > 0:
            for parameter in (self._network.convnet.parameters()):
                parameter.requires_grad = False

        train_dataset = (data_manager.get_dataset(np.arange(self._known_classes,self._total_classes),source="train",mode="train"))

        self.train_loader = (
            build_train_loader(
                dataset=train_dataset,

                dataset_name=(
                    self.args[
                        "dataset"
                    ]
                ),

                batch_size=(
                    self.args[
                        "batch_size"
                    ]
                ),

                num_workers=(
                    self.args[
                        "num_workers"
                    ]
                ),

                seed=(
                    int(
                        self.args[
                            "seed"
                        ]
                    )
                    + self._cur_task
                ),
            )
        )

        test_dataset = (data_manager.get_dataset(np.arange(0,self._total_classes),source="test",mode="test"))

        self.test_loader = (
            build_eval_loader(
                dataset=test_dataset,

                batch_size=(
                    self.args[
                        "batch_size"
                    ]
                ),

                num_workers=(
                    self.args[
                        "num_workers"
                    ]
                ),
            )
        )


        self._train(self.train_loader,self.test_loader)

    def _train(self,train_loader,test_loader):
        self._network.to(self._device)

        if hasattr(self._network,"module"):
            self._network_module_ptr = (self._network.module)
        else:
            self._network_module_ptr = (self._network)

        # =====================================
        # Task 0:
        # train feature extractor from scratch
        # =====================================
        if self._cur_task == 0:
            self._epoch_num = (self.args["init_epochs"])

            optimizer = optim.SGD(
                filter(
                    lambda p:
                        p.requires_grad,

                    self._network
                    .parameters(),
                ),

                momentum=0.9,

                lr=self.args[
                    "init_lr"
                ],

                weight_decay=self.args[
                    "init_weight_decay"
                ],
            )

            scheduler = (
                optim.lr_scheduler
                .CosineAnnealingLR(
                    optimizer=optimizer,

                    T_max=self.args[
                        "init_epochs"
                    ],
                )
            )

            self._train_function(train_loader,test_loader,optimizer,scheduler)

            # Initial cosine classifier
            # weights used by official FeCAM
            # during initial-task Mahalanobis eval.
            self._build_base_protos()

            # Actual class feature means.
            self._build_protos()

        # =====================================
        # Task 1+:
        # backbone frozen, no raw-image
        # optimizer training
        # =====================================
        else:
            self._build_protos()
            # Replace new cosine classifier
            # weights with class prototypes.
            self._update_fc()

        # =====================================
        # Build/update covariance statistics
        # =====================================
        self._update_covariances()

    def _train_function(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self._epoch_num))

        for epoch in prog_bar:
            self._network.train()

            losses = 0.0
            correct = 0
            total = 0

            for (_,inputs,targets) in train_loader:
                inputs = inputs.to(self._device,non_blocking=True)
                targets = targets.to(self._device,dtype=torch.long,non_blocking=True)
                logits = (self._network(inputs)["logits"])
                loss = (F.cross_entropy(logits,targets))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses += (loss.item())

                preds = torch.argmax(logits,dim=1)

                correct += (preds.eq(targets).sum().item())
                total += (targets.size(0))

            scheduler.step()

            train_acc = (100.0 * correct / max(total,1))
            test_acc = (self._compute_accuracy(self._network,test_loader,))
            avg_loss = (losses / len(train_loader))

            self._last_loss = (float(avg_loss))

            info = (
                f"Task {self._cur_task}, "
                f"Epoch {epoch + 1}/"
                f"{self._epoch_num} => "
                f"Loss {avg_loss:.3f}, "
                f"Train_accy "
                f"{train_acc:.2f}, "
                f"Test_accy "
                f"{test_acc:.2f}"
            )

            prog_bar.set_description(info)
            logging.info(info)

    # Prototype extraction
    def _build_base_protos(self):
        for class_idx in range(self._known_classes,self._total_classes):
            class_weight = (self._network.fc.weight[class_idx].detach().clone())
            self._init_protos.append(class_weight)

    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes,self._total_classes):
                (_,_,class_dataset) = (
                    self.data_manager
                    .get_dataset(
                        np.arange(
                            class_idx,
                            class_idx + 1,
                        ),

                        source="train",

                        # IMPORTANT:
                        # no augmentation
                        mode="test",

                        ret_data=True,
                    )
                )


                class_loader = (
                    build_eval_loader(
                        dataset=(
                            class_dataset
                        ),

                        batch_size=(
                            self.args[
                                "batch_size"
                            ]
                        ),

                        num_workers=(
                            self.args[
                                "num_workers"
                            ]
                        ),
                    )
                )

                vectors, _ = (self._extract_vectors(class_loader))
                class_mean = (np.mean(vectors,axis=0))

                class_mean = (torch.from_numpy(class_mean).float().to(self._device))

                self._protos.append(class_mean)

        if (len(self._protos) != self._total_classes):
            raise RuntimeError(
                "FeCAM prototype count "
                "mismatch: "
                f"{len(self._protos)} "
                f"!= "
                f"{self._total_classes}"
            )

    # Cosine classifier
    def _update_fc(self):
        num_new_classes = (self._total_classes - self._known_classes)

        new_protos = torch.stack(self._protos[-num_new_classes:],dim=0).to(self._device)

        if not hasattr(self._network.fc,"fc2"):

            raise RuntimeError(
                "FeCAM incremental "
                "classifier should be "
                "SplitCosineLinear."
            )

        if (self._network.fc.fc2.weight.shape != new_protos.shape):
            raise RuntimeError(
                "FeCAM FC/prototype "
                "shape mismatch: "
                f"{self._network.fc.fc2.weight.shape} "
                f"vs {new_protos.shape}"
            )

        with torch.no_grad():
            self._network.fc.fc2.weight.copy_(
                new_protos
            )

    # Tukey transformation
    def _tukeys_transform(self,x):
        beta = float(self.args["beta"])
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        if beta == 0:
            return torch.log(torch.clamp(x,min=1e-12))

        return torch.pow(
            torch.clamp(x,min=0.0),beta)

    #Covariance shrinkage
    def shrink_cov(self,cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = (cov.clone())
        off_diag.fill_diagonal_(0.0)

        mask = (off_diag != 0.0)

        if mask.any():
            off_diag_mean = (off_diag[mask].mean())

        else:
            off_diag_mean = (cov.new_tensor(0.0))

        identity = torch.eye(
            cov.shape[0],
            dtype=cov.dtype,
            device=cov.device)


        alpha1 = float(self.args["alpha1"])
        alpha2 = float(self.args["alpha2"])

        return (cov + (alpha1 * diag_mean * identity)
            + (alpha2 * off_diag_mean * (1.0 - identity)))

    #Normalize covariance
    def normalize_cov(self):
        if self.args["shrink"]:
            covariances = (self._cov_mat_shrink)

        else:
            covariances = (self._cov_mat)

        normalized = []

        for cov in covariances:
            diagonal = (torch.diagonal(cov))
            sd = torch.sqrt(torch.clamp(diagonal,min=1e-12))

            scale = (sd.unsqueeze(1) * sd.unsqueeze(0))

            normalized_cov = (cov / torch.clamp(scale,min=1e-12))

            normalized.append(normalized_cov)

        return normalized

    #Per-class covariance extraction
    def _compute_new_cov(self):
        with torch.no_grad():

            for class_idx in range(self._known_classes,self._total_classes):
                (_,_,class_dataset) = (
                    self.data_manager
                    .get_dataset(
                        np.arange(
                            class_idx,
                            class_idx + 1,
                        ),

                        source="train",

                        # IMPORTANT:
                        # deterministic feature
                        # extraction.
                        mode="test",

                        ret_data=True,
                    )
                )


                class_loader = (
                    build_eval_loader(
                        dataset=(
                            class_dataset
                        ),

                        batch_size=(
                            self.args[
                                "batch_size"
                            ]
                        ),

                        num_workers=(
                            self.args[
                                "num_workers"
                            ]
                        ),
                    )
                )


                vectors, _ = (self._extract_vectors(class_loader))

                vectors = (torch.from_numpy(vectors).double())

                if self.args["tukey"]:
                    vectors = (self._tukeys_transform(vectors))

                vectors_np = (vectors.cpu().numpy())
                cov = np.cov(vectors_np.T)

                cov = (torch.from_numpy(cov).double())

                # Official FeCAM performs
                # first shrink here.
                if self.args["shrink"]:
                    cov = (self.shrink_cov(cov))

                self._cov_mat.append(cov)


        if (len(self._cov_mat) != self._total_classes):

            raise RuntimeError(
                "FeCAM covariance count "
                "mismatch: "
                f"{len(self._cov_mat)} "
                f"!= "
                f"{self._total_classes}"
            )

    def _update_covariances(self):
        # Untuk canonical FeCAM:
        # full covariance + per-class.
        if not self.args["full_cov"]:

            raise NotImplementedError(
                "Current thesis FeCAM "
                "integration expects "
                "full_cov=True."
            )


        if not self.args["per_class"]:

            raise NotImplementedError(
                "Current thesis FeCAM "
                "integration expects "
                "per_class=True."
            )


        self._compute_new_cov()

        # Rebuild derived covariance
        # representations for ALL classes.
        self._cov_mat_shrink = []
        self._norm_cov_mat = []


        if self.args["shrink"]:

            # Official implementation
            # applies shrinkage a second time.
            for cov in (self._cov_mat):
                self._cov_mat_shrink.append(self.shrink_cov(cov))


        if self.args["norm_cov"]:
            self._norm_cov_mat = (self.normalize_cov())

        expected = (self._total_classes)

        if self.args["norm_cov"]:
            assert (len(self._norm_cov_mat) == expected)

        elif self.args["shrink"]:
            assert (len(self._cov_mat_shrink) == expected)

    #Mahalanobis distance
    def _mahalanobis(self,vectors,class_mean,cov=None):
        if not torch.is_tensor(vectors):
            vectors = torch.as_tensor(vectors,dtype=torch.float32,device=self._device)

        else:
            vectors = vectors.to(self._device,dtype=torch.float32)

        if not torch.is_tensor(class_mean):
            class_mean = (torch.as_tensor(class_mean,dtype=torch.float32,device=self._device))

        else:
            class_mean = (class_mean.to(self._device,dtype=torch.float32))

        if (self.args["tukey"] and self._cur_task > 0):
            class_mean = (self._tukeys_transform(class_mean))

        vectors = F.normalize(vectors,p=2,dim=-1,eps=1e-12)
        class_mean = F.normalize(class_mean,p=2,dim=-1,eps=1e-12)
        x_minus_mu = (vectors - class_mean)

        if cov is None:
            inv_cov = torch.eye(self.feature_dim,dtype=torch.float32,device=self._device)

        else:
            cov = cov.to(self._device,dtype=torch.float32)
            inv_cov = (torch.linalg.pinv(cov)) 

        left = torch.matmul(x_minus_mu,inv_cov)
        distance = torch.sum(left * x_minus_mu,dim=1)

        return (distance.detach().cpu().numpy())

    # Mahalanobis distance for every class
    def _maha_dist(self,vectors):
        vectors = torch.as_tensor(vectors,dtype=torch.float32,device=self._device)

        if (self.args["tukey"] and self._cur_task > 0):
            vectors = (self._tukeys_transform(vectors))

        distances = []

        for class_idx in range(self._total_classes):
            # Official FeCAM initial task:
            # cosine classifier weights +
            # identity covariance.
            if self._cur_task == 0:
                distance = (
                    self._mahalanobis(vectors,self._init_protos[class_idx],cov=None))

            else:
                if self.args["norm_cov"]:
                    covariance = (self._norm_cov_mat[class_idx])
                elif self.args["shrink"]:
                    covariance = (self._cov_mat_shrink[class_idx])
                else:
                    covariance = (self._cov_mat[class_idx])

                distance = (self._mahalanobis(vectors,self._protos[class_idx],covariance))

            distances.append(distance)

        return np.asarray(distances)

    #Evaluation FeCAM
    def _eval_maha(self,loader):
        self._network.eval()
        vectors, y_true = (self._extract_vectors(loader))

        # Same feature normalization
        # used by official FeCAM.
        norms = np.linalg.norm(vectors,axis=1,keepdims=True)
        vectors = (vectors / (norms + 1e-8))
        distances = (self._maha_dist(vectors))
        scores = (distances.T)
        y_pred = np.argsort(scores,axis=1)[:,:self.topk]

        return (y_pred,y_true)

    def eval_task(self):
        # Diagnostic cosine classifier
        y_pred_cnn, y_true_cnn = (self._eval_cnn(self.test_loader))
        cnn_accy = (self._evaluate(y_pred_cnn,y_true_cnn,))

        # Canonical FeCAM:
        # Mahalanobis classifier
        y_pred, y_true = (self._eval_maha(self.test_loader))
        fecam_accy = (self._evaluate(y_pred,y_true,))
        self._last_canonical_y_pred = (np.asarray(y_pred))
        self._last_canonical_y_true = (np.asarray(y_true))
        self._canonical_eval_name = ("fecam_mahalanobis")

        return (cnn_accy,fecam_accy,)