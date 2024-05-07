import time
import shutil
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from types import SimpleNamespace


class Timer:
    def __init__(self) -> None:
        self.last_time = time.time()
    
    def __call__(self) -> str:
        cur_time = time.time()
        delta = cur_time - self.last_time
        self.last_time = cur_time
        return f"Time elapsed: {delta:.3f} s."
    

def save_checkpoint(state, is_best, cfg):
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    torch.save(state, os.path.join(cfg.paths.checkpoint_dir, cfg.paths.checkpoint))
    if is_best:
        shutil.copyfile(
            os.path.join(cfg.paths.checkpoint_dir, cfg.paths.checkpoint),
            os.path.join(cfg.paths.checkpoint_dir, cfg.paths.best_checkpoint)
        )


def number_of_weights(nn):
    trainable_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in nn.parameters())
    return trainable_params, all_params


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


class Loss1(nn.Module):
    name = "RMSE_E"

    def __init__(self, reduction='mean'):
        super(Loss1, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        loss = ((pred - trg) / trg) ** 2
        if self.reduction == 'none':
            return torch.sqrt(loss)
        elif self.reduction == 'sum':
            return torch.sqrt(loss.sum())
        else:
            return torch.sqrt(loss.mean())


class Loss2(nn.Module):
    name = "MAE_E"

    def __init__(self, reduction='mean'):
        super(Loss2, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        loss = torch.abs((pred - trg) / trg)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()


class Loss3(nn.Module):
    name = "MSE"

    def __init__(self, reduction='mean'):
        super(Loss3, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        return F.mse_loss(pred, trg, reduction=self.reduction)


class Loss4(nn.Module):
    name = "MAE"

    def __init__(self, reduction='mean'):
        super(Loss4, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        loss = torch.abs(pred - trg)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()


class Loss5(nn.Module):
    name = "RMSLE"

    def __init__(self, reduction='mean'):
        super(Loss5, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        loss = F.mse_loss(torch.log(pred + 1), torch.log(trg + 1), reduction='none')
        if self.reduction == 'none':
            return torch.sqrt(loss)
        elif self.reduction == 'sum':
            return torch.sqrt(loss.sum())
        else:
            return torch.sqrt(loss.mean())


class Loss6(nn.Module):
    name = "RR" # Relative relation

    def __init__(self, reduction='mean'):
        super(Loss6, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        loss = (pred - trg) / trg
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()


class Loss7(nn.Module):
    name = "None" # Placeholder loss that returns zero

    def __init__(self, reduction='mean'):
        super(Loss7, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        loss = torch.zeros_like(pred)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()
        

class Loss8(nn.Module):
    name = "RMSE"

    def __init__(self, reduction='mean'):
        super(Loss8, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, trg):
        return torch.sqrt(F.mse_loss(pred, trg, reduction=self.reduction))


def get_loss_fn(loss_name, reduction="mean"):
    all_losses = [Loss1, Loss2, Loss3, Loss4, Loss5, Loss6, Loss7, Loss8]
    for LossClass in all_losses:
        if LossClass.name == loss_name:
            return LossClass(reduction=reduction)
    raise RuntimeError(f"Invalid loss function: {loss_name}")


def compute_metrics(model, val_loader, cfg, loss_fns=["RMSE_E", "MAE_E", "RR", "RMSE", "MSE", "MAE"]):
    model.eval()
    
    all_outputs, all_targets = [], []
    
    with torch.no_grad():
        for data, trg in val_loader:
            data, trg = data.to(cfg.device), trg.to(cfg.device)
            output = model(data)
            
            all_outputs.append(output)
            all_targets.append(trg)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = {}
        no_reduce_metrics = {}
        
        for loss_name in loss_fns:
            loss_fn = get_loss_fn(loss_name)
            loss_fn_no_reduce = get_loss_fn(loss_name, "none")

            metrics[f"{loss_name} eng"] = loss_fn(all_outputs[:, 0], all_targets[:, 0]).item()
            if not loss_name.endswith("_E"):
                metrics[f"{loss_name} pos"] = loss_fn(all_outputs[:, 1:3], all_targets[:, 1:3]).item()
            
            no_reduce_metrics[loss_name] = loss_fn_no_reduce(all_outputs[:, 0], all_targets[:, 0]).cpu().numpy()

    return all_outputs, all_targets, metrics, no_reduce_metrics


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    else:
        return obj


def get_experiment_group_name(cfg):
    group_name = f"exp{cfg.experiment_id}"
    if cfg.experiment_id == 1:
        group_name += f"_{cfg.model.tag}_{cfg.data.height}x{cfg.data.width}"
    else:
        raise RuntimeError(f"Unknown experiment_id: {cfg.experiment_id}")
    return group_name
