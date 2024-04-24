import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

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
    def __init__(self, height=15, width=15, n_scales=7, hidden_dim=4, output_positive=True, **kwargs) -> None:
        super(MyModel, self).__init__()

        self.height = height
        self.width = width

        self.output_positive = output_positive

        self.n_scales = n_scales
        self.hidden_dim = hidden_dim
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

        if self.output_positive:
            result = F.softplus(result)
        return result


def get_model(cfg, logger):
    logger.info("Loading the model")
    timer = Timer()

    cfg.paths.checkpoint_dir = os.path.join(
        cfg.paths.checkpoint_dir,
        f"{cfg.model.tag}_{cfg.data.height}x{cfg.data.width}",
    )

    model_params = namespace_to_dict(cfg.model)

    if cfg.model.tag == "SimpleModel":
        model = SimpleModel(
            height=cfg.data.height,
            width=cfg.data.width,
            **model_params,
        )
    elif cfg.model.tag == "LinearModel":
        model = LinearModel(
            height=cfg.data.height,
            width=cfg.data.width,
            **model_params,
        )
    elif cfg.model.tag == "MyModel":
        model = MyModel(
            height=cfg.data.height,
            width=cfg.data.width,
            **model_params,
        )
    elif cfg.model.tag == "MyResnet18":
        model = MyResnet18(
            height=cfg.data.height,
            width=cfg.data.width,
            **model_params,
        )
    else:
        raise RuntimeError(f"Invalid model tag: {cfg.model.tag}")
    
    model = model.to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.optimizer.learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        cfg.training.scheduler.step_size,
        cfg.training.scheduler.gamma,
    )

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
