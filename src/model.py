import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, VisionTransformer

import math

from utils import Timer
from utils import namespace_to_dict

import os


class SimpleModel(nn.Module):
    def __init__(self, height=15, width=15, **kwargs) -> None:
        super(SimpleModel, self).__init__()

        self.height = height
        self.width = width

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

        self.fc1 = nn.Linear(3600, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.flatten()
        return x


class LinearModel(nn.Module):
    def __init__(self, height=15, width=15, output_positive=False, eps=1e-9, **kwargs) -> None:
        super(LinearModel, self).__init__()

        self.height = height
        self.width = width

        self.fc1 = nn.Linear(self.height * self.width, 1)
        self.output_positive = output_positive
        self.eps = eps
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.height * self.width)
        x = self.fc1(x)
        x = x.flatten()
        if self.output_positive:
            x = F.softplus(x)
        return x


class MyModel(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            n_scales=7,
            hidden_dim=4,
            output_positive=True,
            dropout=0.1,
            **kwargs,
    ) -> None:
        super(MyModel, self).__init__()

        self.height = height
        self.width = width

        self.output_positive = output_positive

        self.n_scales = n_scales
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        self.scales = [0.5 ** i for i in range(self.n_scales)]

        self.models = []
        for scale in self.scales:
            self.models.append(self._get_model())

        self.models = nn.ModuleList(self.models)
    
    def _get_model(self):
        layer = nn.Sequential(
            nn.Linear(self.height * self.width, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout_prob),
        )
        return layer

        # return nn.Linear(self.height * self.width, self.hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.height * self.width)
        
        result = None
        for scale, model in zip(self.scales, self.models):
            current = model(x * scale) / scale
            if result is None:
                result = current
            else:
                result = torch.cat((result, current), dim=1)

        result = result.mean(dim=1)
        result = result.flatten()

        if self.output_positive:
            result = F.softplus(result)
        return result


class MyResnet18(nn.Module):
    def __init__(self, height=15, width=15, remove_batch_norm=True, output_positive=True, **kwargs) -> None:
        super(MyResnet18, self).__init__()

        self.height = height
        self.width = width

        self.output_positive = output_positive

        self.model = resnet18(num_classes=1)
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
        
        result = self.model(x)
        result = result.flatten()

        if self.output_positive:
            result = F.softplus(result)
        return result


class MyCNN(nn.Module):
    def __init__(self, height=15, width=15, hidden_dim=100, output_positive=True, **kwargs):
        super(MyCNN, self).__init__()
        self.height = height
        self.width = width
        self.output_positive = output_positive
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self._to_linear = None
        self._mock_forward_pass(height, width)
        
        self.fc1 = nn.Linear(self._to_linear, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def _mock_forward_pass(self, width, height):
        dummy_x = torch.zeros((1, 1, width, height))
        dummy_x = self.pool(F.relu(self.conv1(dummy_x)))
        dummy_x = self.pool(F.relu(self.conv2(dummy_x)))
        dummy_x = self.pool(F.relu(self.conv3(dummy_x)))
        self._to_linear = int(torch.numel(dummy_x))
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self._to_linear)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.flatten()

        if self.output_positive:
            x = F.softplus(x)

        return x


class MyViT(nn.Module):
    def __init__(
            self,
            height=15,
            width=15,
            patch_size=4,
            hidden_dim=64,
            output_positive=True,
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
        self.output_positive = output_positive
        self.model = VisionTransformer(
            image_size=height,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=1,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = x.flatten()
        if self.output_positive:
            x = F.softplus(x)
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
        SimpleModel,
        LinearModel,
        MyModel,
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
