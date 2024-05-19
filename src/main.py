import wandb
import logging
import os
import argparse

from config import load_config
from data import get_data, extract_dimensions
from model import get_model
from utils import number_of_weights
from utils import Timer, save_checkpoint
from utils import get_loss_fn, compute_metrics

import torch


def train_fn(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        criterion_eng,
        criterion_pos,
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
    
    def log_plots(epoch, val_outputs, val_targets, no_reduce_metrics):
        logger.info("Logging plots to wandb")
        for name, value in no_reduce_metrics.items():
            data = [[x, y] for (x, y) in zip(val_targets[:, 0], value)]
            table = wandb.Table(data=data, columns=["Energy", f"{name} Loss"])
            wandb.log({f"scatter_{name}": wandb.plot.scatter(table, "Energy",  f"{name} Loss")}, step=epoch)

    all_val_losses = []

    if cfg.training.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    alpha = cfg.training.eng_loss_weight

    logger.info("=" * 80)

    for epoch in range(1, cfg.training.num_epochs + 1):

        timer = Timer()
        train_total = torch.zeros((1,), device=cfg.device, dtype=torch.float32)
        train_eng = torch.zeros_like(train_total)
        train_pos = torch.zeros_like(train_total)
        num_batches = 0

        logger.info(f"Epoch: {epoch}/{cfg.training.num_epochs} || LR: {get_lr(optimizer):.6f}")

        model.train()
        for data, trg in train_loader:
            num_batches += 1
            optimizer.zero_grad()

            data = data.to(cfg.device)
            trg = trg.to(cfg.device)

            if cfg.training.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(data)
                    loss_eng = criterion_eng(output[:, 0], trg[:, 0])
                    loss_pos = criterion_pos(output[:, 1:3], trg[:, 1:3])
                    loss = alpha * loss_eng + (1 - alpha) * loss_pos
                
                train_total += loss.detach()
                train_eng += loss_eng.detach()
                train_pos += loss_pos.detach()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss_eng = criterion_eng(output[:, 0], trg[:, 0])
                loss_pos = criterion_pos(output[:, 1:3], trg[:, 1:3])
                loss = alpha * loss_eng + (1 - alpha) * loss_pos

                train_total += loss.detach()
                train_eng += loss_eng.detach()
                train_pos += loss_pos.detach()

                loss.backward()

                optimizer.step()

            if (num_batches % cfg.logging.log_freq == 0) or (num_batches == len(train_loader)):
                # logger.info(f"Batch: {num_batches}/{len(train_loader)} || Train total: {train_total.item() / num_batches:0.4f} || {timer()}")
                logger.info(f"Batch: {num_batches}/{len(train_loader)} || "
                             f"Train total: {train_total.item() / num_batches:0.4f} || "
                             f"Train eng: {train_eng.item() / num_batches:0.4f} || "
                             f"Train pos: {train_pos.item() / num_batches:0.4f} || "
                             f"{timer()}")
                
        train_total = (train_total / num_batches).item()
        train_eng = (train_eng / num_batches).item()
        train_pos = (train_pos / num_batches).item()
        
        val_total = torch.zeros((1,), device=cfg.device, dtype=torch.float32)
        val_eng = torch.zeros_like(val_total)
        val_pos = torch.zeros_like(val_total)
        model.eval()
        with torch.no_grad():
            for data, trg in val_loader:
                data = data.to(cfg.device)
                trg = trg.to(cfg.device)

                if cfg.training.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = model(data)
                        loss_eng = criterion_eng(output[:, 0], trg[:, 0])
                        loss_pos = criterion_pos(output[:, 1:3], trg[:, 1:3])
                        loss = alpha * loss_eng + (1 - alpha) * loss_pos

                        val_total += loss.detach()
                        val_eng += loss_eng.detach()
                        val_pos += loss_pos.detach()
                else:
                    output = model(data)
                    loss_eng = criterion_eng(output[:, 0], trg[:, 0])
                    loss_pos = criterion_pos(output[:, 1:3], trg[:, 1:3])
                    loss = alpha * loss_eng + (1 - alpha) * loss_pos

                    val_total += loss.detach()
                    val_eng += loss_eng.detach()
                    val_pos += loss_pos.detach()
        
        val_total = (val_total / len(val_loader)).item()
        val_eng = (val_eng / len(val_loader)).item()
        val_pos = (val_pos / len(val_loader)).item()

        is_best = (len(all_val_losses) == 0) or (val_total < min(all_val_losses))

        all_val_losses.append(val_total)

        val_outputs, val_targets, metrics, no_reduce_metrics = compute_metrics(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
        )

        # logger.info(f"Val loss: {val_loss:0.4f} || {pretty_metrics(metrics)}")

        logger.info(f"Val total: {val_total:0.4f} || "
                f"Val eng: {val_eng:0.4f} || "
                f"Val pos: {val_pos:0.4f}")
        
        logger.info(f"Val metrics: {pretty_metrics(metrics)}")
        
        if wandb.run is not None:
            wandb.log({
                "train_total": train_total,
                "train_eng": train_eng,
                "train_pos": train_pos,
                "val_total": val_total,
                "val_eng": val_eng,
                "val_pos": val_pos,
                "lr": get_lr(optimizer),
                **metrics,
            }, step=epoch)
            
            if (cfg.logging.log_plot_freq != -1) and ((epoch == cfg.training.num_epochs) or (epoch % cfg.logging.log_plot_freq == 0)):
                log_plots(epoch, val_outputs, val_targets, no_reduce_metrics)

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            state={
                "num_epochs": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "val_total": val_total,
                "val_eng": val_eng,
                "val_pos": val_pos,
            },
            is_best=is_best,
            cfg=cfg,
        )

        logger.info("=" * 80)


def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Training script")
        parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
        return parser.parse_args()

    args = parse_arguments()

    cfg, cfg_dict = load_config(args.config)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = cfg.logging.info_prints

    height, width = extract_dimensions(cfg.paths.data_file)

    cfg.data.height = height
    cfg.data.width = width

    if cfg.model.positive_pos:
        assert cfg.data.normalize_position, "Model always outputs positive values while target positions are not positive"

    if cfg.logging.wandb:
        wandb.init(
            project="ECAL optimization",
            config=cfg_dict
        )
    
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_data(cfg, logger)

    model, optimizer, scheduler = get_model(cfg, logger)

    trainable_params, all_params = number_of_weights(model)

    logger.info(f"Number of trainable parameters: {trainable_params} || Number of all parameters: {all_params}")

    # if wandb.run is not None:
    #     wandb.watch(model, log="all")

    criterion_eng = get_loss_fn(cfg.training.loss_fn_eng)
    criterion_pos = get_loss_fn(cfg.training.loss_fn_pos)

    train_fn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_eng=criterion_eng,
        criterion_pos=criterion_pos,
        cfg=cfg,
        logger=logger,
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    
    main()
