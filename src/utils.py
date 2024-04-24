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
    return sum(p.numel() for p in nn.parameters() if p.requires_grad)


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

    def __init__(self):
        super(Loss1, self).__init__()
    
    def forward(self, pred, trg):
        return torch.sqrt(torch.mean(((pred - trg) / trg) ** 2))


class Loss2(nn.Module):
    name = "MAE_E"

    def __init__(self):
        super(Loss2, self).__init__()
    
    def forward(self, pred, trg):
        return torch.mean(torch.abs((pred - trg) / trg))


class Loss3(nn.Module):
    name = "MSE"

    def __init__(self):
        super(Loss3, self).__init__()
    
    def forward(self, pred, trg):
        return F.mse_loss(pred, trg)


class Loss4(nn.Module):
    name = "MAE"

    def __init__(self):
        super(Loss4, self).__init__()
    
    def forward(self, pred, trg):
        return torch.mean(torch.abs(pred - trg))


class Loss5(nn.Module):
    name = "RMSLE"

    def __init__(self):
        super(Loss5, self).__init__()
    
    def forward(self, pred, trg):
        return torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(trg + 1)))


class Loss6(nn.Module):
    name = "RR" # Relative relation

    def __init__(self):
        super(Loss6, self).__init__()
    
    def forward(self, pred, trg):
        return torch.mean((pred - trg) / trg)


def get_loss_fn(loss_name):
    all_losses = [Loss1, Loss2, Loss3, Loss4, Loss5, Loss6]
    for LossClass in all_losses:
        if LossClass.name == loss_name:
            return LossClass()
    raise RuntimeError(f"Invalid loss function: {loss_name}")


def compute_metrics(model, val_loader, cfg, loss_fns=["RMSE_E", "MAE_E", "RMSLE", "RR"]):
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
        
        for loss_name in loss_fns:
            loss_fn = get_loss_fn(loss_name)
            metrics[loss_name] = loss_fn(all_outputs, all_targets).item()
 
    return metrics


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    else:
        return obj
