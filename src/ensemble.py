import wandb
import logging
import os
import re
import copy

from config import load_config
from model import get_model
from data import get_data
from utils import get_loss_fn, compute_metrics
from utils import Timer

import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models) -> None:
        super(Ensemble, self).__init__()
        
        self.models = models
    
    def forward(self, x):
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, device=x.device)
        for model in self.models:
            result += model(x)
        result /= len(self.models)
        return result


def get_ensemble(cfg, logger):
    mem_cfg = copy.deepcopy(cfg)

    cfg.paths.checkpoint_dir = os.path.join(
        cfg.paths.checkpoint_dir,
        f"{cfg.model.tag}_{cfg.data.height}x{cfg.data.width}",
    )
    all_models = []

    def get_runs(path):
        tmp = os.listdir(path)
        res = []
        for run_name in tmp:
            if os.path.isdir(os.path.join(path, run_name)):
                run_id = re.findall(r'\d+', run_name)[0]
                run_id = int(run_id)
                res.append((run_id, run_name))
        res.sort(key=lambda x: x[0])
        res = res[-cfg.model.ensemble.num_models:]
        return [x[1] for x in res]
    
    runs = get_runs(cfg.paths.checkpoint_dir)
    logger.info(f"Ensemble of runs: {[os.path.join(cfg.paths.checkpoint_dir, run) for run in runs]}")

    for run in runs:
        logger.info("=" * 40)
        model, optimizer, scheduler = get_model(mem_cfg, logger, create_subdirs=False)
        if cfg.model.ensemble.use_best:
            checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, run, cfg.paths.best_checkpoint)
        else:
            checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, run, cfg.paths.checkpoint)
        logger.info(f"Loading model checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model'])

        logger.info(f"Val loss: {checkpoint['val_loss']:0.4f}")

        all_models.append(model)

    logger.info("=" * 40)

    model = Ensemble(all_models)
    return model


def eval_fn(
        train_loader,
        val_loader,
        model,
        criterion,
        cfg,
        logger,
    ):
    
    def pretty_metrics(metrics):
        result = ""
        for name, value in metrics.items():
            if len(result) > 0:
                result += " || "
            result += f"{name}: {value:0.4f}"
        return result

    logger.info("=" * 80)

    timer = Timer()

    model.eval()
    with torch.no_grad():
        train_loss = torch.zeros((1,), device=cfg.device, dtype=torch.float32)

        for data, trg in train_loader:
            data = data.to(cfg.device)
            trg = trg.to(cfg.device)

            output = model(data)
            loss = criterion(output, trg)

            train_loss += loss.detach()
        
        train_loss = (train_loss / len(train_loader)).item()
        
        val_loss = torch.zeros((1,), device=cfg.device, dtype=torch.float32)
    
        for data, trg in val_loader:
            data = data.to(cfg.device)
            trg = trg.to(cfg.device)

            if cfg.training.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(data)
                    val_loss += criterion(output, trg).detach()
            else:
                output = model(data)
                val_loss += criterion(output, trg).detach()
        
        val_loss = (val_loss / len(val_loader)).item()
    
    logger.info(f"Train loss: {train_loss:0.4f} || Val loss: {val_loss:0.4f}")

    val_outputs, val_targets, metrics, no_reduce_metrics = compute_metrics(
        model=model,
        val_loader=val_loader,
        cfg=cfg,
    )

    logger.info(f"Val metrics: {pretty_metrics(metrics)}")

    logger.info("=" * 80)


if __name__ == "__main__":
    cfg, cfg_dict = load_config()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = cfg.logging.info_prints
    
    cfg.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_data(cfg, logger)

    model = get_ensemble(cfg, logger)

    criterion = get_loss_fn(cfg.training.loss_fn)

    eval_fn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        cfg=cfg,
        logger=logger,
    )
