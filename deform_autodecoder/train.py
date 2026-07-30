from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deform_autodecoder.dataloader import build_dataloaders, set_seed
from deform_autodecoder.model import DeformAutoencoder, deform_reconstruction_loss


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_deformation_tensor(path):
    tensor = torch.load(path, map_location="cpu")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor)}.")

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(
            f"Expected tensor shape [N,H,W] or [N,1,H,W], got {tuple(tensor.shape)}."
        )

    if tensor.shape[1] != 1:
        raise ValueError(
            f"Expected a single-channel tensor [N,1,H,W], got {tuple(tensor.shape)}."
        )

    if tensor.shape[-2:] != (240, 240):
        raise ValueError(
            f"Expected spatial size (240, 240), got {tuple(tensor.shape[-2:])}."
        )

    return tensor.float()


def build_loader(tensor_path, batch_size, num_workers, shuffle):
    tensor = load_deformation_tensor(tensor_path)
    dataset = TensorDataset(tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def _make_image_logs(images, recon, split_name, epoch, max_images):
    image_logs = []
    count = min(max_images, images.shape[0])

    for idx in range(count):
        gt = images[idx].detach().cpu().float().numpy()
        pred = recon[idx].detach().cpu().float().numpy()

        if gt.ndim == 3 and gt.shape[0] in {1, 3}:
            gt = gt.transpose(1, 2, 0)
        if pred.ndim == 3 and pred.shape[0] in {1, 3}:
            pred = pred.transpose(1, 2, 0)

        if gt.ndim == 3 and gt.shape[-1] == 1:
            gt = gt[..., 0]
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]

        image_logs.append(
            wandb.Image(
                gt,
                caption=f"{split_name}/epoch_{epoch}/sample_{idx}_ground_truth",
            )
        )
        image_logs.append(
            wandb.Image(
                pred,
                caption=f"{split_name}/epoch_{epoch}/sample_{idx}_reconstruction",
            )
        )

    return image_logs


def _make_image_logs_with_step(
    images,
    recon,
    split_name,
    epoch,
    step,
    max_images,
):
    image_logs = []
    count = min(max_images, images.shape[0])

    for idx in range(count):
        gt = images[idx].detach().cpu().float().numpy()
        pred = recon[idx].detach().cpu().float().numpy()

        if gt.ndim == 3 and gt.shape[0] in {1, 3}:
            gt = gt.transpose(1, 2, 0)
        if pred.ndim == 3 and pred.shape[0] in {1, 3}:
            pred = pred.transpose(1, 2, 0)

        if gt.ndim == 3 and gt.shape[-1] == 1:
            gt = gt[..., 0]
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]

        image_logs.append(
            wandb.Image(
                gt,
                caption=(
                    f"{split_name}/epoch_{epoch}/step_{step}/"
                    f"sample_{idx}_ground_truth"
                ),
            )
        )
        image_logs.append(
            wandb.Image(
                pred,
                caption=(
                    f"{split_name}/epoch_{epoch}/step_{step}/"
                    f"sample_{idx}_reconstruction"
                ),
            )
        )

    return image_logs


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    runtime_cfg,
    epoch,
    split_name,
    compute_gradients=True,
    update_parameters=False,
):
    if update_parameters:
        model.train()
    else:
        model.eval()

    loss_cfg = runtime_cfg["loss"]
    wandb_cfg = runtime_cfg["wandb"]
    total_loss = 0.0
    total_pixel = 0.0
    total_edge = 0.0
    num_batches = 0
    image_logs = None

    context = torch.enable_grad() if compute_gradients else torch.no_grad()
    with context:
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch} [{split_name}]",
            leave=False,
        )
        for step, batch in enumerate(progress, start=1):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch["image"]

            images = images.to(device, non_blocking=True)
            recon = model(images)
            losses = deform_reconstruction_loss(
                recon,
                images,
                pixel_loss_type=loss_cfg["pixel_loss_type"],
                edge_weight=float(loss_cfg["edge_weight"]),
            )

            if update_parameters:
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()

            progress.set_postfix(
                loss=f"{losses['loss'].item():.6f}",
                pixel=f"{losses['pixel_loss'].item():.6f}",
                edge=f"{losses['edge_loss'].item():.6f}",
            )

            total_loss += losses["loss"].item()
            total_pixel += losses["pixel_loss"].item()
            total_edge += losses["edge_loss"].item()
            num_batches += 1

            if wandb_cfg["enabled"]:
                global_step = (epoch - 1) * len(loader) + step
                if update_parameters and step % int(runtime_cfg["training"]["log_every"]) == 0:
                    wandb.log(
                        {
                            f"{split_name}/batch_loss": losses["loss"].item(),
                            f"{split_name}/batch_pixel_loss": losses["pixel_loss"].item(),
                            f"{split_name}/batch_edge_loss": losses["edge_loss"].item(),
                            "epoch": epoch,
                            f"{split_name}/step": global_step,
                        }
                    )

                should_log_train_images = (
                    split_name == "train"
                    and update_parameters
                    and int(wandb_cfg["log_train_images_every_batches"]) > 0
                    and step % int(wandb_cfg["log_train_images_every_batches"]) == 0
                )
                if should_log_train_images:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/step": global_step,
                            "train/reconstructions_step": _make_image_logs_with_step(
                                images=images,
                                recon=recon,
                                split_name=split_name,
                                epoch=epoch,
                                step=step,
                                max_images=int(wandb_cfg["num_images"]),
                            ),
                        }
                    )

            should_log_images = (
                image_logs is None
                and wandb_cfg["enabled"]
                and epoch % int(wandb_cfg["log_images_every"]) == 0
            )
            if should_log_images:
                image_logs = _make_image_logs(
                    images=images,
                    recon=recon,
                    split_name=split_name,
                    epoch=epoch,
                    max_images=int(wandb_cfg["num_images"]),
                )

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "pixel_loss": total_pixel / max(num_batches, 1),
        "edge_loss": total_edge / max(num_batches, 1),
    }
    return metrics, image_logs


def main():
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    data_cfg = config["data"]
    training_cfg = config["training"]

    source = data_cfg["source"]

    device_name = training_cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = DeformAutoencoder(
        input_channels=int(config["preprocess"]["num_channels"]),
        latent_channels=int(config["model"]["latent_channels"]),
        output_activation=config["model"]["output_activation"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    if source == "tensor":
        train_path = data_cfg["train_tensor_path"]
        if not train_path:
            raise ValueError("config.yaml data.train_tensor_path is empty.")

        val_path = data_cfg["val_tensor_path"] or train_path
        test_path = data_cfg["test_tensor_path"] or val_path
        train_loader = build_loader(
            train_path,
            batch_size=int(training_cfg["batch_size"]),
            num_workers=int(training_cfg["num_workers"]),
            shuffle=True,
        )
        val_loader = build_loader(
            val_path,
            batch_size=int(training_cfg["batch_size"]),
            num_workers=int(training_cfg["num_workers"]),
            shuffle=False,
        )
        test_loader = build_loader(
            test_path,
            batch_size=int(training_cfg["batch_size"]),
            num_workers=int(training_cfg["num_workers"]),
            shuffle=False,
        )
    elif source == "dataloader":
        set_seed(int(data_cfg["dataloader"]["split"]["seed"]))
        normal_loaders, _ = build_dataloaders(config)
        train_loader = normal_loaders["train"]
        val_loader = normal_loaders["val"]
        test_loader = normal_loaders["test"]
    else:
        raise ValueError("data.source must be either 'tensor' or 'dataloader'.")

    save_dir = Path(training_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    num_epochs = int(training_cfg["num_epochs"])
    runtime_cfg = {
        "loss": config["loss"],
        "training": training_cfg,
        "wandb": config["wandb"],
    }

    wandb_cfg = config["wandb"]
    if wandb_cfg["enabled"]:
        run_name = wandb_cfg["run_name"] or (
            f"deform_ae_{source}_{config['model']['output_activation']}"
        )
        wandb.init(
            project=wandb_cfg["project"],
            name=run_name,
            config=config,
        )

    best_path = save_dir / "best.pt"

    for epoch in range(1, num_epochs + 1):
        train_metrics, train_images = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            runtime_cfg,
            epoch=epoch,
            split_name="train",
            compute_gradients=True,
            update_parameters=True,
        )
        val_metrics, val_images = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            runtime_cfg,
            epoch=epoch,
            split_name="val",
            compute_gradients=True,
            update_parameters=False,
        )
        test_metrics, test_images = run_epoch(
            model,
            test_loader,
            optimizer,
            device,
            runtime_cfg,
            epoch=epoch,
            split_name="test",
            compute_gradients=False,
            update_parameters=False,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"test_loss={test_metrics['loss']:.6f}"
        )

        if wandb_cfg["enabled"]:
            log_payload = {
                "epoch": epoch,
                "train/epoch_loss": train_metrics["loss"],
                "train/epoch_pixel_loss": train_metrics["pixel_loss"],
                "train/epoch_edge_loss": train_metrics["edge_loss"],
                "val/epoch_loss": val_metrics["loss"],
                "val/epoch_pixel_loss": val_metrics["pixel_loss"],
                "val/epoch_edge_loss": val_metrics["edge_loss"],
                "test/epoch_loss": test_metrics["loss"],
                "test/epoch_pixel_loss": test_metrics["pixel_loss"],
                "test/epoch_edge_loss": test_metrics["edge_loss"],
            }
            if train_images is not None:
                log_payload["train/reconstructions"] = train_images
            if val_images is not None:
                log_payload["val/reconstructions"] = val_images
            if test_images is not None:
                log_payload["test/reconstructions"] = test_images
            wandb.log(log_payload)

        latest_path = save_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            latest_path,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                best_path,
            )

    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_test_metrics, final_test_images = run_epoch(
            model,
            test_loader,
            optimizer,
            device,
            runtime_cfg,
            epoch=num_epochs,
            split_name="test_best",
            compute_gradients=False,
            update_parameters=False,
        )
        print(
            "best checkpoint test "
            f"loss={final_test_metrics['loss']:.6f}"
        )
        if wandb_cfg["enabled"]:
            log_payload = {
                "best/test_loss": final_test_metrics["loss"],
                "best/test_pixel_loss": final_test_metrics["pixel_loss"],
                "best/test_edge_loss": final_test_metrics["edge_loss"],
            }
            if final_test_images is not None:
                log_payload["best/test_reconstructions"] = final_test_images
            wandb.log(log_payload)

    if wandb_cfg["enabled"]:
        wandb.finish()


if __name__ == "__main__":
    main()
