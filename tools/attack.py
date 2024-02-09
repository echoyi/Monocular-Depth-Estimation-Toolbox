from typing import Tuple, Dict

import numpy as np
import torch
# from models.downstream import LinearClassifier, Segmenter, NormalizationWrapper
from depth.models.depther.encoder_decoder import DepthEncoderDecoder

from typing import Callable
from torchvision import transforms
import mmcv


class Loss:
    name: str
    loss_func = torch.nn.Module
    loss: torch.nn.Module

    def __call__(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, true)

class L2(Loss):
    name = "L2"
    loss_func = torch.nn.MSELoss
    loss = torch.nn.MSELoss()

class Mismatch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        true_logits = pred[torch.arange(len(pred)), true].clone()
        pred[torch.arange(len(pred)), true] = -torch.inf
        target_logits = pred.max(dim=1).values
        return (target_logits - true_logits).sum()


class MismatchLoss(Loss):
    name = "mismatch"
    loss = Mismatch()
    loss_func = Mismatch


class Attack:
    name: str

    def __init__(
        self,
        loss: Loss,
        num_steps: int,
        lr: float,
        eps_budget: float,
        mean: tuple = None,
        std: tuple = None,
    ):
        self.loss = loss
        self.num_steps = num_steps
        self.lr = lr
        self.eps_budget = eps_budget
        self.left_boundary = 0
        self.right_boundary = 1
        self.mean = mean
        self.std = std

        self._normalize = transforms.Normalize(mean, std)
        self._denormalize = transforms.Normalize(
            -np.array(mean) / np.array(std), 1 / np.array(std)
        )

    def run(self, images, model, **kwargs):
        raise NotImplementedError


class PGD(Attack):
    name: str
    loss: Loss
    num_steps: int
    lr: float
    eps_budget: float
    optimizer: torch.optim.Optimizer = None
    delta: torch.Tensor = None

    def _init_delta_optim(self, images: torch.Tensor):
        raise NotImplementedError

    def _clamp_delta(self, images):
        # makes sure the delta is within budget and we don't get out of bounds of
        # [lb; rb] for images + delta
        self.delta.data = (
            torch.clamp_(
                images + self.delta.data,
                min=self.left_boundary,
                max=self.right_boundary,
            )
            - images
        )
        self.delta.data = torch.clamp_(
            self.delta.data, min=-self.eps_budget, max=self.eps_budget
        )

    def _optimize_delta(self):
        raise NotImplementedError

    def run(
        self,
        images: torch.Tensor,
        model: DepthEncoderDecoder,
        precomputed_original_representations: torch.Tensor = None,
        override_delta: torch.Tensor = None,
        return_step_by_step: bool = False,
        steps_to_save: int = 100,
        get_representation: Callable = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self._denormalize(images)
        if get_representation is None:
            get_representation = model.extract_feat
        if override_delta is None:  # useful for step-by-step evaluation of pgd
            self._init_delta_optim(images)
        if precomputed_original_representations is None:
            with torch.no_grad():
                original_representations = get_representation(self._normalize(images))
        else:
            original_representations = precomputed_original_representations

        if return_step_by_step:  # useful for step-by-step evaluation of pgd
            step_by_step_representations = []

        for idx in range(self.num_steps):
            adv_representations = get_representation(
                self._normalize(images + self.delta)
            )
            self.loss(adv_representations, original_representations).backward()

            self._optimize_delta()
            self._clamp_delta(images)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.delta.grad.zero_()

            if return_step_by_step and not idx % int(
                self.num_steps / steps_to_save
            ):  # don't save all by default
                step_by_step_representations.append(
                    adv_representations.clone().detach().unsqueeze(1).cpu()
                )

        adv_images = self._normalize(images + self.delta)
        if return_step_by_step:
            return (
                adv_images.detach(),
                torch.concatenate(step_by_step_representations, dim=1),
            )
        return adv_images.detach(), adv_representations.detach()

class PGD_Sign(PGD):
    name = "sign"

    # Adapted from https://github.com/PKU-ML/DYNACL/blob/master/train_DynACL.py
    def _init_delta_optim(self, images: torch.Tensor):
        self.delta = torch.rand_like(images) * self.eps_budget * 2 - self.eps_budget
        self.delta = self.delta.detach().requires_grad_(True)

    def _optimize_delta(self):
        self.delta.data += torch.sign(self.delta.grad) * self.lr



class DownstreamPGD(PGD):
    name = "downstream"
    loss = MismatchLoss()

    def __init__(self, num_steps, lr, eps_budget, loss=None, mean=None, std=None):
        super().__init__(self.loss, num_steps, lr, eps_budget, mean, std)

    def _optimize_delta(self):
        self.delta.data += self.lr * torch.sign(self.delta.grad)

    def _init_delta_optim(self, images: torch.Tensor):
        self.delta = torch.zeros_like(images).uniform_(
            -self.eps_budget, self.eps_budget
        )

        self.delta = self.delta.detach().requires_grad_(True)

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        model: DepthEncoderDecoder,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self._denormalize(images)
        self._init_delta_optim(images)

        for _ in range(self.num_steps):
            logits = model(self._normalize(images + self.delta))
            self.loss(logits, labels).backward()

            self._optimize_delta()
            self._clamp_delta(images)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.delta.grad.zero_()

        adv_images = self._normalize(images + self.delta)
        success = torch.argmax(model(adv_images), dim=1) != labels
        return adv_images.detach(), success.detach()
    

class SegPGD(PGD):
    name = "SegPGD"

    def run(
        self,
        model: DepthEncoderDecoder,
        images: torch.Tensor,
        labels: torch.Tensor,
        device=None,
        targeted=False,
    ):
        # model for volumetric image segmentation
        # images: [B,C,H,W]. B=BatchSize, C=Number-of-Channels,  H=Height,  W=Width
        # labels: [B,1,H,W] (in integer form)

        print(
            f"SegPGD: alpha={self.lr} , eps={self.eps_budget} , steps={self.num_steps} , targeted={targeted}\n"
        )

        images = self._denormalize(images).clone().detach().to(device)  #  [B,C,H,W]
        labels = labels.clone().detach().to(device)  #  [B,H,W]
        adv_images = images.clone().detach()

        # starting at a uniformly random point
        self.delta = torch.empty_like(adv_images).uniform_(
            -self.eps_budget, self.eps_budget
        )

        self._clamp_delta(images)

        adv_images = adv_images + self.delta

        for i in range(self.num_steps):
            adv_images.requires_grad = True
            # adv_images[B,C,H,W] --> adv_logits[B,NumClass,H,W]
            adv_logits = model(self._normalize(adv_images))
            # [B,NumClass,H,W,D] --> [B,H,W]
            pred_labels = torch.argmax(adv_logits, dim=1)
            # correctly classified voxels  [B,H,W]
            correct_voxels = labels == pred_labels
            # wrongly classified voxels    [B,H,W]
            wrong_voxels = labels != pred_labels
            # [B,NumClass,H,W] -->  [NumClass,B,H,W]
            adv_logits = adv_logits.permute(1, 0, 2, 3)
            # calculate loss
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss_correct = loss_fn(
                adv_logits[:, correct_voxels].permute(1, 0),
                labels[correct_voxels],
            )
            loss_wrong = loss_fn(
                adv_logits[:, wrong_voxels].permute(1, 0),
                labels[wrong_voxels],
            )
            lmbda = i / (2 * self.num_steps)
            loss = (1 - lmbda) * loss_correct + lmbda * loss_wrong
            if targeted:
                loss = -1 * loss

            # update adversarial images
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]
            adv_images = adv_images.detach() + self.lr * grad.sign()
            self.delta = adv_images - images
            self._clamp_delta(images)
            adv_images = images + self.delta
        print(
            f"Adversarial Loss: {round(loss.item(), 5):3.5f}",
        )

        return self._normalize(adv_images)


#change the miou and other stuff with other metrics   
# def adversarial_segmenter_accuracy_miou(
#     attack: Attack,
#     encoder: Encoder,
#     data_loader: DataLoader,
#     device,
#     segmenter: Segmenter,
#     n_labels: int,
#     downstream: bool = False,
# ) -> [float, float]:
#     metric = evaluate.load("mean_iou")
#     for idx, (images, labels) in tqdm(enumerate(data_loader)):
#         if downstream:
#             adv_img = attack.run(
#                 images=images.to(device),
#                 model=segmenter,
#                 labels=labels,
#                 device=device,
#             )
#         else:
#             adv_img, _ = attack.run(
#                 images=images.to(device),
#                 model=encoder,
#                 precomputed_original_representations=None,
#                 get_representations=encoder.get_patch_embeddings,
#                 return_step_by_step=False,  # TODO returns error otherwise; fix it
#             )
#         with torch.no_grad():
#             predictions = segmenter.forward(adv_img).argmax(dim=1).cpu()
#         metric.add_batch(
#             predictions=predictions,
#             references=labels,
#         )
#         if (idx + 1) * data_loader.batch_size >= 1000:
#             break
#     metrics = metric.compute(num_labels=n_labels, ignore_index=0, reduce_labels=False)
#     return metrics["mean_accuracy"], metrics["mean_iou"]


def evalute_attack_accuracy(model, loader):
    loss = L2.loss_func
    num_steps = 5
    lr = 0.05
    eps_budget = 0.03137254901960784
    mean = 0
    std = 0
    model.eval()
    results = []
    dataset = loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = loader.batch_sampler
    for batch_indices, data in zip(loader_indices, loader):
        model(return_loss=False, **data)
        # model.extract_feat()

        with torch.no_grad():
            result_depth = model(return_loss=False, **data)

    # for idx, (images, labels) in enumerate(loader):
        # pass