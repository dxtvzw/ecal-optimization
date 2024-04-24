import wandb
import logging

from config import load_config
from data import get_data
from model import get_model
from utils import number_of_weights, get_model_size
from utils import Timer, save_checkpoint
from utils import get_loss_fn, compute_metrics

import torch


def train_fn(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        criterion,
        cfg,
        logger,
    ):

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    def pretty_metrics(metrics):
        result = ""
        for name, value in metrics.items():
            if len(result) > 0:
                result += " || "
            result += f"{name}: {value:0.4f}"
        return result

    all_val_losses = []

    logger.info("=" * 80)

    for epoch in range(1, cfg.training.num_epochs + 1):

        timer = Timer()
        train_loss = torch.zeros((1,), device=device, dtype=torch.float32)
        num_batches = 0

        logger.info(f"Epoch: {epoch}/{cfg.training.num_epochs}")

        model.train()
        for data, trg in train_loader:
            num_batches += 1
            optimizer.zero_grad()

            data = data.to(device)
            trg = trg.to(device)

            output = model(data)
            loss = criterion(output, trg)

            train_loss += loss.detach()

            loss.backward()

            optimizer.step()

            if (num_batches % cfg.logging.log_freq == 0) or (num_batches == len(train_loader)):
                logger.info(f"Batch: {num_batches}/{len(train_loader)} || Train loss: {train_loss.item() / num_batches:0.4f} || {timer()}")
        
        train_loss = (train_loss / num_batches).item()
        
        val_loss = torch.zeros((1,), device=device, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            for data, trg in val_loader:
                data = data.to(device)
                trg = trg.to(device)

                output = model(data)
                val_loss += criterion(output, trg).detach()
        
        val_loss /= len(val_loader)

        val_loss = val_loss.item()

        is_best = (len(all_val_losses) == 0) or (val_loss < min(all_val_losses))

        all_val_losses.append(val_loss)

        metrics = compute_metrics(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
        )

        logger.info(f"Val loss: {val_loss:0.4f} || {pretty_metrics(metrics)}")

        if wandb.run is not None:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": get_lr(optimizer),
                **metrics,
            }, step=epoch)

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            state={
                "num_epochs": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "val_loss": val_loss,
            },
            is_best=is_best,
            cfg=cfg,
        )

        logger.info("=" * 80)


if __name__ == "__main__":
    cfg, cfg_dict = load_config()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = cfg.logging.info_prints

    if cfg.logging.wandb:
        wandb.init(
            project="ECAL optimization",
            config=cfg_dict
        )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg.device = device

    train_loader, val_loader = get_data(cfg, logger)

    model, optimizer, scheduler = get_model(cfg, logger)

    if wandb.run is not None:
        wandb.watch(model, log="all")

    criterion = get_loss_fn(cfg.training.loss_fn)

    train_fn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        cfg=cfg,
        logger=logger,
    )
