import copy
import logging
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.distributions import (
    MultivariateNormal,
)

from models.base import BaseLearner

from utils.inc_net import (
    get_convnet,
)

from thesis.data.loaders import (
    build_train_loader,
    build_eval_loader,
)

class EFCNet(nn.Module):

    def __init__(self,args):
        super().__init__()

        self.convnet = get_convnet(args,pretrained=False)
        self.heads = nn.ModuleList()

    @property
    def feature_dim(self):
        return self.convnet.out_dim


    def add_head(self,num_classes):
        self.heads.append(
            nn.Linear(
                self.feature_dim,
                num_classes,
                bias=False,
            )
        )


    def extract_vector(self,x):
        return self.convnet(x)["features"]

    def forward(self,x):
        backbone_out = (self.convnet(x))

        features = (backbone_out["features"])
        outputs = {
            task_id:
                head(features)
            for task_id, head in enumerate(self.heads)
        }

        return (outputs,features)

    def freeze_all(self):
        for parameter in (self.parameters()):
            parameter.requires_grad = False

        self.eval()

    def freeze_bn(self):
        for module in (self.convnet.modules()):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()

                for parameter in (module.parameters()):
                    parameter.requires_grad = False

class EFC(BaseLearner):

    def __init__(self,args):
        super().__init__(args)

        self._task_sizes = []

        self._last_loss_cls = 0.0
        self._last_loss_efm = 0.0
        self._last_loss_proto = 0.0

        self.args = args

        self._network = (EFCNet(args))

        self._old_network = None

        self.previous_efm = None

        self.efc_lambda = float(args.get("efc_lamb",10.0))

        self.efc_damping = float(args.get("efc_damping",0.1))

        self.sigma_proto_update = float(args.get("efc_protoupdate",0.2))

        self.proto_batch_size = int(args.get("efc_protobatchsize",64))

        # Gaussian prototype state
        self._prototypes = []
        self._prototype_covariances = []
        self._prototype_labels = []

        self._gaussians = {}

        self._running_prototypes = []
        self._running_covariances = []

        # Previous current-task batch
        # used by PR-ACE.
        self._previous_batch_samples = None
        self._previous_batch_targets = None

        self._rotation_classifier = None

    # Helpers
    def _task_offset(self,task_id):
        return sum(self._task_sizes[:task_id])

    def _rescale_targets(self,targets,task_id):
        return (targets - self._task_offset(task_id))

    def _concat_logits(self,outputs):
        return torch.cat([outputs[task_id] for task_id in sorted(outputs.keys())],dim=1)

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            _, images, targets = batch

        elif len(batch) == 2:
            images, targets = batch

        else:
            raise ValueError(
                "Unexpected EFC batch "
                f"format with {len(batch)} "
                "elements."
            )

        return (images,targets)

    # Self rotation
    def _compute_rotations(self,images,targets,task_id):
        if task_id != 0:
            raise ValueError(
                "EFC self-rotation is "
                "only used on Task 0."
            )


        num_classes = (self._task_sizes[0])

        local_targets = (self._rescale_targets(targets,task_id))

        rotated_images = torch.stack(
            [
                torch.rot90(
                    images,
                    k=k,
                    dims=(2, 3),
                )

                for k in range(
                    1,
                    4,
                )
            ],
            dim=1,
        )


        rotated_images = (rotated_images.flatten(0,1))

        rotated_targets = torch.stack(
            [
                (
                    local_targets * 3
                    + k
                    + num_classes
                    - 1
                )

                for k in range(
                    1,
                    4,
                )
            ],
            dim=1,
        ).reshape(-1)


        return (rotated_images,rotated_targets)

    # Auxiliary classifier
    def _build_rotation_classifier(self):
        num_classes = (self._task_sizes[0])

        self._rotation_classifier = (nn.Linear(self._network.feature_dim,num_classes * 3).to(self._device))

    def _efm_loss(self,features,old_features):
        if self.previous_efm is None:
            raise RuntimeError(
                "Previous EFM is not "
                "available."
            )


        efm = (self.previous_efm.to(device=features.device,dtype=features.dtype))

        identity = torch.eye(efm.shape[0],device=features.device,dtype=features.dtype)

        matrix_reg = (self.efc_lambda * efm + self.efc_damping * identity)

        delta = (features - old_features)

        loss = torch.einsum("bi,ij,bj->b",delta,matrix_reg,delta).mean()

        return loss

    def _compute_efm(self,loader):
        self._network.eval()

        feature_dim = (self._network.feature_dim)

        efm = torch.zeros((feature_dim,feature_dim),device=self._device,dtype=torch.float32)

        # Official implementation ignores
        # the final incomplete batch.
        num_full_batches = (len(loader.dataset) // loader.batch_size)

        if num_full_batches == 0:
            raise RuntimeError(
                "Not enough samples to "
                "compute EFM."
            )

        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if (batch_idx >= num_full_batches):
                    break

                images, _ = (self._unpack_batch(batch))
                images = images.to(self._device,non_blocking=True)

                outputs, features = (self._network(images))
                logits = (self._concat_logits(outputs))

                probabilities = (F.softmax(logits,dim=1))

                weight_matrix = torch.cat([head.weight for head in (self._network.heads)],dim=0)

                num_classes = (logits.shape[1])

                identity = torch.eye(num_classes,device=self._device,dtype=logits.dtype)

                # d log-softmax / d logits
                derivative = (identity.unsqueeze(0) - probabilities.unsqueeze(1))

                # Closed-form Jacobian
                # with respect to feature vector.
                jacobian = torch.matmul(derivative,weight_matrix)

                # Sum_c p(c|x)
                # J_c^T J_c
                efm_per_sample = (torch.einsum("bc,bcd,bce->bde",probabilities,jacobian,jacobian))
                efm += (efm_per_sample.sum(dim=0))

                total_samples += (images.shape[0])


        efm = (efm / total_samples)

        # Numerical symmetrization.
        efm = (0.5 * (efm + efm.T))

        return efm.detach()

    # Gaussian prototype registration
    def _register_gaussian(self,label,features):
        if features.ndim != 2:
            raise ValueError(
                "Prototype features must "
                "have shape [N, D]."
            )


        if features.shape[0] < 2:
            raise ValueError(
                "At least two samples "
                "are required for a "
                "covariance matrix."
            )

        features = (features.detach().float().cpu())
        mean = (features.mean(dim=0))

        covariance = (torch.cov(features.T))
        covariance = (0.5 * (covariance + covariance.T))

        eps = 1e-5

        covariance_safe = (covariance + eps * torch.eye(covariance.shape[0],dtype=covariance.dtype))
        gaussian = (MultivariateNormal(mean,covariance_matrix=(covariance_safe)))

        self._prototypes.append(mean)
        self._prototype_covariances.append(covariance)
        self._prototype_labels.append(int(label))

        self._gaussians[int(label)] = gaussian

    def _compute_task_prototypes(self,loader,task_id):
        self._network.eval()

        features_all = []
        labels_all = []


        with torch.no_grad():
            for batch in loader:
                images, labels = (self._unpack_batch(batch))

                images = images.to(self._device,non_blocking=True)

                labels = labels.to(dtype=torch.long)

                _, features = (self._network(images))

                features_all.append(features.detach().cpu())

                labels_all.append(labels.cpu())

        features_all = torch.cat(features_all,dim=0)
        labels_all = torch.cat(labels_all,dim=0)

        start_class = (self._task_offset(task_id))
        end_class = (start_class + self._task_sizes[task_id])

        for class_idx in range(start_class,end_class):
            mask = (labels_all == class_idx)
            class_features = (features_all[mask])

            if (class_features.shape[0] < 2):
                raise RuntimeError(
                    "Not enough feature "
                    f"samples for class "
                    f"{class_idx}."
                )

            self._register_gaussian(class_idx,class_features)

        self._running_prototypes = [prototype.clone() for prototype in (self._prototypes)]

        self._running_covariances = [covariance.clone() for covariance in (self._prototype_covariances)]

    def _sample_old_prototypes(self,task_id,num_samples):
        if task_id <= 0:
            raise ValueError(
                "Prototype replay is only "
                "used after Task 0."
            )

        num_old_classes = (self._task_offset(task_id))
        old_labels = [int(label) for label in (self._prototype_labels) if int(label) < num_old_classes]

        if len(old_labels) == 0:
            raise RuntimeError(
                "No old Gaussian prototypes "
                "available for PR-ACE."
            )

        if num_samples <= 0:
            return (torch.empty(0,self._network.feature_dim),torch.empty(0,dtype=torch.long))

        # Shuffle one copy of all old classes.
        permutation = torch.randperm(len(old_labels))

        shuffled = [old_labels[index] for index in (permutation.tolist())]

        # Official EFC repeats labels when
        # requested prototype batch is larger
        # than number of old classes.
        if num_samples > len(shuffled):

            repeats = int(np.ceil(num_samples / len(shuffled)))

            selected_labels = (shuffled * repeats)[:num_samples]

        else:
            selected_labels = (shuffled[:num_samples])

        # Group by class before Gaussian sampling.
        selected_labels = sorted(selected_labels)
        samples = []
        labels = []

        for class_idx in sorted(set(selected_labels)):
            count = (selected_labels.count(class_idx))

            if class_idx not in (self._gaussians):
                raise RuntimeError(
                    "Missing Gaussian for "
                    f"class {class_idx}."
                )

            class_samples = self._gaussians[class_idx].sample((count,))
            samples.append(class_samples)

            labels.extend([class_idx] * count)

        samples = torch.cat(samples,dim=0)

        labels = torch.tensor(labels,dtype=torch.long)

        # Official code shuffles generated
        # prototype samples before returning.
        permutation = torch.randperm(samples.shape[0])

        samples = samples[permutation]
        labels = labels[permutation]

        return (samples,labels)

    def _update_current_batch_buffer(self,images,targets):
        # Engineering detail:
        # keep it on CPU between iterations
        # to avoid retaining unnecessary VRAM.
        self._previous_batch_samples = (images.detach().cpu().clone())
        self._previous_batch_targets = (targets.detach().cpu().clone())

    def _sample_current_batch_buffer(self,task_id,batch_size):
        if (self._previous_batch_samples is None):
            return (self.proto_batch_size,None,None)

        num_old_classes = (self._task_offset(task_id))

        num_current_classes = (self._task_sizes[task_id])

        total_classes = (num_old_classes + num_current_classes)

        # This follows the balancing intent
        # of official Buffer.sample().
        if total_classes >= batch_size:
            sampled_class_slots = (torch.randperm(total_classes)[:batch_size])

        else:
            sampled_class_slots = (torch.randint(low=0,high=total_classes,size=(batch_size,)))

        num_proto = int((sampled_class_slots < num_old_classes).sum().item())
        num_current_replay = (batch_size - num_proto)
        available = min(num_current_replay,len(self._previous_batch_targets))
        previous_images = (self._previous_batch_samples[:available])
        previous_targets = (self._previous_batch_targets[:available])

        # If an incomplete previous batch cannot
        # provide enough current-task samples,
        # fill the missing slots with prototypes.
        num_proto += (num_current_replay - available)
        return (num_proto,previous_images,previous_targets)

    # Helper taks 0 rotation logits
    def _append_rotation_logits(self,outputs,features):
        if self._rotation_classifier is None:
            raise RuntimeError(
                "Rotation classifier "
                "has not been created."
            )

        outputs = dict(outputs)
        rotation_logits = (self._rotation_classifier(features))
        outputs[0] = torch.cat([outputs[0],rotation_logits,],dim=1)

        return outputs

    # PR ACE Criterion
    def _train_criterion(self,outputs,targets,task_id,features,old_features,proto_to_samples,current_batch_size):
        zero = (features.new_zeros(()))
        loss_cls = zero
        loss_efm = zero
        loss_proto = zero

        # =====================================
        # Task 0
        # =====================================
        if task_id == 0:
            logits = (self._concat_logits(outputs))
            loss_cls = (F.cross_entropy(logits,targets))

            return (loss_cls,loss_efm,loss_proto)

        # =====================================
        # Task 1+
        # =====================================

        if old_features is None:
            raise RuntimeError(
                "old_features required "
                "for EFC Task 1+."
            )

        # -------------------------------------
        # EFM feature consolidation
        # Only original current batch.
        # -------------------------------------
        loss_efm = (
            self._efm_loss(features[:current_batch_size],old_features))

        # -------------------------------------
        # Current-class classification
        # -------------------------------------
        current_targets = (self._rescale_targets(targets[:current_batch_size],task_id))

        loss_cls = (F.cross_entropy(outputs[task_id][:current_batch_size],current_targets))

        # -------------------------------------
        # Gaussian prototype replay
        # -------------------------------------
        replay_logits = []
        replay_targets = []

        if proto_to_samples > 0:
            (proto_features,proto_labels) = (self._sample_old_prototypes(task_id,proto_to_samples))

            proto_features = (
                proto_features.to(self._device,dtype=features.dtype))

            proto_labels = (proto_labels.to(self._device,dtype=torch.long))

            proto_logits = torch.cat([head(proto_features) for head in (self._network.heads)],dim=1)

            replay_logits.append(proto_logits)
            replay_targets.append(proto_labels)

        # -------------------------------------
        # Previous batch of CURRENT task
        # -------------------------------------
        current_replay_logits = (self._concat_logits(outputs)[current_batch_size:])
        current_replay_targets = (targets[current_batch_size:])

        if (current_replay_logits.shape[0] > 0):
            replay_logits.append(current_replay_logits)
            replay_targets.append(current_replay_targets)

        if len(replay_logits) > 0:
            replay_logits = torch.cat(replay_logits,dim=0)
            replay_targets = torch.cat(replay_targets,dim=0)

            loss_proto = (F.cross_entropy(replay_logits,replay_targets))

        return (loss_cls,loss_efm,loss_proto)

    def _collect_old_new_features(self,loader):
        if self._old_network is None:
            raise RuntimeError(
                "Old network is required "
                "to compute feature drift."
            )

        self._network.eval()
        self._old_network.eval()

        new_features_all = []
        old_features_all = []

        with torch.no_grad():
            for batch in loader:
                images, _ = (self._unpack_batch(batch))
                images = images.to(self._device,non_blocking=True)

                _, new_features = (self._network(images))
                _, old_features = (self._old_network(images))

                new_features_all.append(new_features.detach().cpu())
                old_features_all.append(old_features.detach().cpu())

        new_features_all = torch.cat(new_features_all,dim=0)
        old_features_all = torch.cat(old_features_all,dim=0)

        return (new_features_all,old_features_all)

    # Compute prototype drift
    def _compute_drift(self,new_features,old_features):
        if self.previous_efm is None:
            raise RuntimeError(
                "previous_efm is required "
                "for prototype drift."
            )

        if len(self._running_prototypes) == 0:
            raise RuntimeError(
                "No running prototypes "
                "available."
            )

        new_features = (new_features.float().to(self._device))
        old_features = (old_features.float().to(self._device))

        efm = (self.previous_efm.float().to(self._device))

        # Feature displacement:
        # current model - previous model
        delta_y = (new_features - old_features)

        running_prototypes = (torch.stack(self._running_prototypes,dim=0).float().to(self._device))

        num_prototypes = (running_prototypes.shape[0])
        num_samples = (new_features.shape[0])

        distance = torch.zeros((num_prototypes,num_samples),device=self._device)

        for i in range(num_prototypes):
            diff = (old_features - running_prototypes[i].unsqueeze(0))

            # Negative EFM-weighted
            # squared distance.
            distance[i] = -torch.einsum("bi,ij,bj->b",diff,efm,diff)

        dist_min = (distance.min())
        dist_max = (distance.max())

        denom = torch.clamp(dist_max - dist_min,min=1e-12)

        scaled_distance = (distance - dist_min) / denom

        sigma = float(self.sigma_proto_update)

        if sigma <= 0:
            raise ValueError(
                "sigma_proto_update "
                "must be > 0 or -1 "
                "to disable update."
            )

        weights = torch.exp(scaled_distance / (2.0 * sigma**2))
        weights = (weights / torch.clamp(weights.sum(dim=1,keepdim=True),min=1e-12))

        displacement = torch.zeros((num_prototypes,self._network.feature_dim),device=self._device)

        for i in range(num_prototypes):
            displacement[i] = (weights[i].unsqueeze(1) * delta_y).sum(dim=0)

        return (displacement.detach().cpu())

    # Gaussian Helper
    def _set_gaussian(self,label,mean,covariance):
        label = int(label)

        mean = (mean.detach().float().cpu())

        covariance = (
            covariance
            .detach()
            .float()
            .cpu()
        )

        covariance_safe = (
            covariance
            + 1e-5
            * torch.eye(
                covariance.shape[0],
                dtype=covariance.dtype,
            )
        )

        self._gaussians[
            label
        ] = MultivariateNormal(
            mean,
            covariance_matrix=(
                covariance_safe
            ),
        )

    def _update_old_prototypes_with_drift(self,drift):
        num_old_prototypes = len(self._prototypes)
        if (
            drift.shape[0]
            != num_old_prototypes
        ):
            raise RuntimeError(
                "Prototype drift count "
                "mismatch: "
                f"{drift.shape[0]} vs "
                f"{num_old_prototypes}"
            )

        for i in range(
            num_old_prototypes
        ):

            updated_mean = (
                self._prototypes[i]
                + drift[i]
            )

            covariance = (
                self
                ._prototype_covariances[
                    i
                ]
            )

            label = (
                self
                ._prototype_labels[
                    i
                ]
            )

            self._prototypes[i] = (
                updated_mean
                .detach()
                .cpu()
            )

            self._set_gaussian(
                label=label,
                mean=updated_mean,
                covariance=covariance,
            )

        # Official EFC then refreshes
        # running_proto.
        self._running_prototypes = [
            prototype.clone()

            for prototype
            in self._prototypes
        ]

    # Statistic loader helper
    def _build_task_statistics_loader(self,task_id):
        start_class = (
            self._task_offset(
                task_id
            )
        )

        end_class = (
            start_class
            + self._task_sizes[
                task_id
            ]
        )

        dataset = (
            self.data_manager
            .get_dataset(
                np.arange(
                    start_class,
                    end_class,
                ),
                source="train",

                # No random train
                # augmentation.
                mode="test",
            )
        )

        loader = (
            build_eval_loader(
                dataset=dataset,

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

        return loader

    # Combine all (using all of the provided methods)
    def _post_task_statistics(self,task_id):
        statistics_loader = (
            self._build_task_statistics_loader(
                task_id
            )
        )

        # =====================================
        # 1. Update OLD prototypes
        # =====================================
        if (
            task_id > 0
            and
            self.sigma_proto_update
            != -1
        ):

            (
                new_features,
                old_features,
            ) = (
                self
                ._collect_old_new_features(
                    statistics_loader
                )
            )

            drift = (
                self._compute_drift(
                    new_features,
                    old_features,
                )
            )

            self._update_old_prototypes_with_drift(
                drift
            )

            self._last_drift_norm = float(
                torch.linalg.vector_norm(
                    drift
                ).item()
            )

        else:

            self._last_drift_norm = 0.0

        # =====================================
        # 2. Compute EFM for CURRENT task
        # =====================================
        new_efm = (self._compute_efm(statistics_loader))

        self.previous_efm = (new_efm.detach().clone())

        # Diagnostics only
        singular_values = (torch.linalg.svdvals(self.previous_efm.cpu()))

        self._efm_rank = int(torch.linalg.matrix_rank(self.previous_efm.cpu()).item())
        self._efm_largest_sv = float(singular_values[0].item())

        # =====================================
        # 3. Add CURRENT task prototypes
        # =====================================
        self._compute_task_prototypes(statistics_loader,task_id)

        # =====================================
        # Sanity checks
        # =====================================
        expected_classes = sum(self._task_sizes[:task_id + 1])

        if (len(self._prototypes) != expected_classes):
            raise RuntimeError(
                "EFC prototype count "
                "mismatch after task "
                f"{task_id}: "
                f"{len(self._prototypes)} "
                f"!= {expected_classes}"
            )

        if (len(self._prototype_covariances) != expected_classes):
            raise RuntimeError("EFC covariance count ""mismatch.")

        if (len(self._gaussians) != expected_classes):
            raise RuntimeError("EFC Gaussian count " "mismatch.")

    def after_task(self):
        self._known_classes = (self._total_classes)

        if self.args.get("save_checkpoints",True,):
            checkpoint_prefix = (
                self.args.get(
                    "checkpoint_prefix",
                    "{}_{}_{}".format(
                        self.args["model_name"],
                        self.args["init_cls"],
                        self.args["increment"],),))
            self.save_checkpoint(checkpoint_prefix)

    # Helper to freeze old model
    def _build_old_network(self):
        self._old_network = (copy.deepcopy(self._network))

        for parameter in (self._old_network.parameters()):
            parameter.requires_grad = False

        self._old_network.to(self._device)
        self._old_network.eval()

    # EFC Optimizer
    def _build_optimizer(self,task_id):
        if task_id == 0:
            parameters = [p for p in (self._network.parameters()) if p.requires_grad]

            if (self._rotation_classifier is not None):
                parameters += [p for p in (self._rotation_classifier.parameters()) if p.requires_grad]

            optimizer = (torch.optim.SGD(parameters,lr=0.1,momentum=0.9,weight_decay=5e-4))

            scheduler = (
                torch.optim.lr_scheduler
                .MultiStepLR(
                    optimizer,
                    milestones=[
                        80,
                        120,
                        150,
                    ],
                    gamma=0.1,
                )
            )

        else:
            # Official EFC freezes BN
            # after Task 0.
            self._network.freeze_bn()

            backbone_params = [p for p in (self._network.convnet.parameters()) if p.requires_grad]

            # EFC ORIGINAL:
            # old + new heads are optimized.
            head_params = [p for p in (self._network.heads.parameters()) if p.requires_grad]

            optimizer = (
                torch.optim.Adam(
                    [
                        {
                            "params":
                                head_params,
                            "lr":
                                1e-4,
                        },
                        {
                            "params":
                                backbone_params,
                            "lr":
                                1e-5,
                        },
                    ],
                    weight_decay=2e-4,
                )
            )


            # Official implementation uses
            # effectively fixed LR after
            # Task 0.
            scheduler = (
                torch.optim.lr_scheduler
                .MultiStepLR(
                    optimizer,

                    milestones=[
                        1000,
                        2000,
                    ],

                    gamma=0.1,
                )
            )

        return (optimizer,scheduler)

    def _train_one_epoch(self,task_id,train_loader,optimizer):
        # Official EFC:
        # Task 0 = train()
        # Task 1+ = eval(), so BN stats stay frozen.
        if task_id == 0:
            self._network.train()
            if (self._rotation_classifier is not None):
                self._rotation_classifier.train()

        else:
            self._network.eval()

        total_loss = 0.0
        total_cls = 0.0
        total_efm = 0.0
        total_proto = 0.0

        total_samples = 0
        correct = 0

        for batch in train_loader:
            (images,targets) = (self._unpack_batch(batch))

            images = images.to(self._device,non_blocking=True)
            targets = targets.to(self._device,dtype=torch.long,non_blocking=True)

            current_batch_size = (images.shape[0])

            # ==================================
            # Task 0 self-rotation
            # ==================================
            if task_id == 0:
                (rotated_images,rotated_targets) = (self._compute_rotations(images,targets,task_id=0))

                images_for_model = torch.cat([images,rotated_images],dim=0)

                targets_for_loss = torch.cat([targets,rotated_targets],dim=0)
                old_features = None
                proto_to_samples = 0

            # ==================================
            # Task 1+
            # ==================================
            else:

                # OLD model sees only original
                # current mini-batch.
                with torch.no_grad():
                    (_,old_features) = (self._old_network(images))

                (proto_to_samples,replay_images,replay_targets) = (
                    self
                    ._sample_current_batch_buffer(
                        task_id,
                        self.args[
                            "batch_size"
                        ],
                    )
                )


                if (replay_images is not None and replay_images.shape[0] > 0):
                    replay_images = (replay_images.to(self._device,dtype=images.dtype))

                    replay_targets = (replay_targets.to(self._device,dtype=torch.long))

                    images_for_model = (torch.cat([images,replay_images],dim=0))

                    targets_for_loss = (torch.cat([targets,replay_targets],dim=0))

                else:
                    images_for_model = (images)
                    targets_for_loss = (targets)

            # ==================================
            # Current network forward
            # ==================================
            (outputs,features) = (self._network(images_for_model))

            if task_id == 0:
                outputs = (self._append_rotation_logits(outputs,features))

            # ==================================
            # Criterion
            # ==================================
            (loss_cls,loss_efm,loss_proto) = (
                self._train_criterion(
                    outputs=outputs,

                    targets=targets_for_loss,

                    task_id=task_id,

                    features=features,

                    old_features=(
                        old_features
                    ),

                    proto_to_samples=(
                        proto_to_samples
                    ),

                    current_batch_size=(
                        current_batch_size
                    ),
                )
            )

            loss = (loss_cls + loss_efm + loss_proto)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ==================================
            # Accuracy diagnostic
            # only ORIGINAL current samples
            # ==================================
            with torch.no_grad():
                if task_id == 0:
                    logits_diag = (outputs[0][:current_batch_size,:self._task_sizes[0]])
                    targets_diag = (targets)

                else:
                    logits_diag = (outputs[task_id][:current_batch_size])
                    targets_diag = (self._rescale_targets(targets,task_id))

                predictions = (logits_diag.argmax(dim=1))
                correct += (predictions.eq(targets_diag).sum().item())
                total_samples += (current_batch_size)

            # ==================================
            # Current-task one-batch buffer
            # ==================================
            if task_id > 0:
                self._update_current_batch_buffer(images,targets)

            total_loss += (float(loss.detach()))
            total_cls += (float(loss_cls.detach()))
            total_efm += (float(loss_efm.detach()))
            total_proto += (float(loss_proto.detach()))

        num_batches = len(train_loader)

        self._last_loss = (total_loss / num_batches)
        self._last_loss_cls = (total_cls / num_batches)
        self._last_loss_efm = (total_efm / num_batches)
        self._last_loss_proto = (total_proto / num_batches)

        train_accuracy = (100.0 * correct / max(total_samples,1))

        return {
            "loss":
                self._last_loss,
            "loss_cls":
                self._last_loss_cls,
            "loss_efm":
                self._last_loss_efm,
            "loss_proto":
                self._last_loss_proto,
            "train_acc":
                train_accuracy,
        }

    # Training orchestration
    def _train_task(self,task_id,train_loader):
        (optimizer,scheduler) = (self._build_optimizer(task_id))

        if task_id == 0:
            num_epochs = int(self.args["epochs_first_task"])
        
        else:
            num_epochs = int(self.args["epochs_next_task"])

        for epoch in range(num_epochs):
            metrics = (self._train_one_epoch(task_id,train_loader,optimizer))

            scheduler.step()

            logging.info(
                "Task %d, Epoch %d/%d => "
                "Loss %.3f, "
                "Loss_cls %.3f, "
                "Loss_efm %.3f, "
                "Loss_proto %.3f, "
                "Train_accy %.2f",
                task_id,
                epoch + 1,
                num_epochs,
                metrics["loss"],
                metrics["loss_cls"],
                metrics["loss_efm"],
                metrics["loss_proto"],
                metrics["train_acc"],
            )

        return metrics

    def incremental_train(self,data_manager):
        self.data_manager = (data_manager)

        next_task = (self._cur_task + 1)

        if next_task == 0:
            expected_known = 0

        else:
            expected_known = (data_manager.get_accumulate_tasksize( next_task - 1))

        if (self._known_classes != expected_known):
            raise RuntimeError(
                "EFC task state mismatch. "
                f"Task={next_task}, "
                f"known={self._known_classes}, "
                f"expected={expected_known}."
            )

        task_size = (data_manager.get_task_size(next_task))

        # =====================================
        # Important:
        # copy model BEFORE adding new head.
        # =====================================
        self._build_old_network()

        self._cur_task = (next_task)

        self._task_sizes.append(task_size)

        self._total_classes = (self._known_classes + task_size)

        # =====================================
        # Add current-task classifier
        # =====================================
        self._network.add_head(task_size)

        self._network.to(self._device)

        # =====================================
        # Task-0 self-rotation classifier
        # =====================================
        if self._cur_task == 0:
            self._build_rotation_classifier()

        else:
            self._rotation_classifier = None

        # Reset temporary current-task buffer.
        self._previous_batch_samples = None
        self._previous_batch_targets = None

        # =====================================
        # Actual optimizer-training dataset
        # =====================================
        current_classes = np.arange(self._known_classes,self._total_classes)

        train_dataset = (data_manager.get_dataset(current_classes,source="train",mode="train"))

        self.train_loader = (
            build_train_loader(
                dataset=train_dataset,
                dataset_name=(self.args["dataset"]),
                batch_size=(self.args["batch_size"]),
                num_workers=(self.args["num_workers"]),
                seed=(int(self.args["seed"]) + self._cur_task)
            )
        )

        # =====================================
        # Test all encountered classes
        # =====================================
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

        logging.info("EFC learning on %d-%d",self._known_classes,self._total_classes)

        # =====================================
        # Optimize current task
        # =====================================
        self._train_task(self._cur_task,self.train_loader)

        # =====================================
        # Drift + EFM + Gaussian prototypes
        # =====================================
        self._post_task_statistics(self._cur_task)

    def _eval_multihead(self,loader):
        self._network.eval()

        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch in loader:
                (images,targets) = (self._unpack_batch(batch))

                images = images.to(self._device,non_blocking=True)
                outputs, _ = (self._network(images))
                logits = (self._concat_logits(outputs))
                k = min(self.topk,logits.shape[1])

                predictions = (torch.topk(logits,k=k,dim=1,largest=True,sorted=True)[1])

                y_pred.append(predictions.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        return (np.concatenate(y_pred,axis=0),np.concatenate(y_true,axis=0))

    def eval_task(self):
        (y_pred,y_true) = (self._eval_multihead(self.test_loader))
        cnn_accy = (self._evaluate(y_pred,y_true))

        return (cnn_accy,None)
