import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, VisionTransformer

import math

from utils import Timer
from utils import namespace_to_dict

import os


class CustomSoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positive_eng, positive_pos):
        ctx.save_for_backward(x)
        ctx.positive_eng = positive_eng
        ctx.positive_pos = positive_pos
        
        if positive_eng:
            x[:, 0] = torch.nn.functional.softplus(x[:, 0])
        if positive_pos:
            x[:, 1:] = torch.nn.functional.softplus(x[:, 1:])
        
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        if ctx.positive_eng:
            grad_input[:, 0] *= torch.nn.functional.sigmoid(x[:, 0])
        if ctx.positive_pos:
            grad_input[:, 1:] *= torch.nn.functional.sigmoid(x[:, 1:])
        
        return grad_input, None, None


def custom_softplus(x, positive_eng=False, positive_pos=False):
    return CustomSoftplusFunction.apply(x, positive_eng, positive_pos)


class SumModel(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            positive_eng=True,
            positive_pos=True,
            **kwargs
        ) -> None:
        super(SumModel, self).__init__()

        self.height = height
        self.width = width
        self.positive_eng = positive_eng
        self.positive_pos = positive_pos

        self.fc1 = nn.Linear(1, 3)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.height * self.width)

        x = self.fc1(x.sum(dim=1).unsqueeze(1))
        
        x = custom_softplus(x, self.positive_eng, self.positive_pos)
        return x


class SimpleModel(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            positive_eng=True,
            positive_pos=True,
            **kwargs
        ) -> None:
        super(SimpleModel, self).__init__()

        self.height = height
        self.width = width
        self.positive_eng = positive_eng
        self.positive_pos = positive_pos

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.fc1 = nn.Linear(1600, 3)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = custom_softplus(x, self.positive_eng, self.positive_pos)
        return x


class LinearModel(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            positive_eng=True,
            positive_pos=True,
            eps=1e-9,
            **kwargs
        ) -> None:
        super(LinearModel, self).__init__()

        self.height = height
        self.width = width

        self.fc1 = nn.Linear(self.height * self.width, 3)
        self.positive_eng = positive_eng
        self.positive_pos = positive_pos
        self.eps = eps
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.height * self.width)
        x = self.fc1(x)
        x = custom_softplus(x, self.positive_eng, self.positive_pos)
        return x


class MyResnet18(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            remove_batch_norm=True,
            positive_eng=True,
            positive_pos=True,
            **kwargs
        ) -> None:
        super(MyResnet18, self).__init__()

        self.height = height
        self.width = width

        self.positive_eng = positive_eng
        self.positive_pos = positive_pos

        self.model = resnet18(num_classes=3)
        self.model.conv1 = nn.Conv2d(1, 64, 1, bias=False)

        if remove_batch_norm:
            self._remove_batch_norms(self.model)

    def _remove_batch_norms(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                identity_layer = nn.Identity()
                setattr(module, name, identity_layer)
            else:
                self._remove_batch_norms(child)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        x = self.model(x)

        x = custom_softplus(x, self.positive_eng, self.positive_pos)
        return x


class MyCNN(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            hidden_dim=100,
            num_layers=3,
            scale_mult=8,
            positive_eng=True,
            positive_pos=True,
            **kwargs
        ):
        super(MyCNN, self).__init__()
        self.height = height
        self.width = width
        self.positive_eng = positive_eng
        self.positive_pos = positive_pos

        last_dim = 1
        self.conv_layers = []
        for i in range(num_layers):
            self.conv_layers.append(nn.Conv2d(last_dim, last_dim * scale_mult, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            last_dim *= scale_mult
        
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self._to_linear = None
        self._mock_forward_pass(height, width)
        
        self.fc1 = nn.Linear(self._to_linear, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def _mock_forward_pass(self, width, height):
        dummy_x = torch.zeros((1, 1, width, height))
        dummy_x = self.conv_layers(dummy_x)
        dummy_x = self.pool(dummy_x)
        self._to_linear = int(torch.numel(dummy_x))
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        x = self.conv_layers(x)
        x = self.pool(x)
        
        x = x.view(-1, self._to_linear)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # print("LOOOK", batch_size, x.size())

        x = custom_softplus(x, self.positive_eng, self.positive_pos)

        return x


class MyViT(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            patch_size=4,
            hidden_dim=64,
            positive_eng=True,
            positive_pos=True,
            num_layers=4,
            num_heads=8,
            mlp_dim=256,
            dropout=0.01,
            attention_dropout=0.01,
            **kwargs
        ):
        super(MyViT, self).__init__()
        self.height = height
        self.width = width
        self.positive_eng = positive_eng
        self.positive_pos = positive_pos

        self.model = VisionTransformer(
            image_size=height,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=3,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = custom_softplus(x, self.positive_eng, self.positive_pos)
        return x


# Credits to: https://github.com/TalSchuster/pytorch-transformers/tree/master
class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


# Credits to: https://github.com/TalSchuster/pytorch-transformers/tree/master
class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def get_model(cfg, logger, create_subdirs=True):
    logger.info("Loading the model")
    logger.info(f"Current PyTorch seed: {torch.seed()}")
    timer = Timer()

    cfg.paths.checkpoint_dir = os.path.join(
        cfg.paths.checkpoint_dir,
        f"{cfg.model.tag}_{cfg.data.height}x{cfg.data.width}",
    )
    if create_subdirs:
        os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
        
    cur_run_id = len(list(os.walk(cfg.paths.checkpoint_dir)))
    cfg.paths.checkpoint_dir = os.path.join(
        cfg.paths.checkpoint_dir,
        f"run_{cur_run_id}",
    )

    model_params = namespace_to_dict(cfg.model)

    model_types = [
        SumModel,
        SimpleModel,
        LinearModel,
        MyResnet18,
        MyCNN,
        MyViT,
    ]

    model = None
    for model_type in model_types:
        if model_type.__name__ == cfg.model.tag:
            model = model_type(**model_params)
    
    if model is None:
        raise RuntimeError(f"Invalid model tag: {cfg.model.tag}")
    
    model = model.to(cfg.device)

    if cfg.training.optimizer.tag == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    elif cfg.training.optimizer.tag == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    else:
        raise RuntimeError(f"Invalid optimizer tag: {cfg.optimizer.tag}")

    if cfg.training.scheduler.tag == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.training.scheduler.step_size,
            cfg.training.scheduler.gamma,
        )
    elif cfg.training.scheduler.tag == "WarmupCosineSchedule":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=cfg.training.scheduler.warmup_steps,
            t_total=cfg.training.num_epochs,
        )
    elif cfg.training.scheduler.tag == "WarmupLinearSchedule":
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=cfg.training.scheduler.warmup_steps,
            t_total=cfg.training.num_epochs,
        )
    else:
        raise RuntimeError(f"Invalid scheduler tag: {cfg.scheduler.tag}")

    checkpoint_path = None

    if cfg.model.use_checkpoint == "last" and os.path.isfile(os.path.join(cfg.paths.checkpoint_dir, cfg.paths.checkpoint)):
        checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, cfg.paths.checkpoint)
    elif cfg.model.use_checkpoint == "best" and os.path.isfile(os.path.join(cfg.paths.checkpoint_dir, cfg.paths.best_checkpoint)):
        checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, cfg.paths.best_checkpoint)

    if checkpoint_path is not None:
        logger.info(f"Loading model checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])


    logger.info(f"Successfully loaded the model || {timer()}")

    return model, optimizer, scheduler
