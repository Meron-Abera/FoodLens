"""Fine-tuning loop."""
import time

import pandas as pd
import torch

from src.models.architectures import build_efficientnet_b0, load_trained_efficientnet_b0


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Runs one pass over `loader`; trains if an optimizer is given, otherwise only evaluates.

    Returns:
        (mean loss, accuracy) over the pass.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def fine_tune(model, head, loss_fn, train_loader, val_loader, checkpoint, config, device):
    """Fine-tunes a pretrained model and keeps the epoch with the best validation accuracy.

    The pretrained layers and the new head get separate learning rates (`lr_backbone`,
    `lr_head`), so the head learns quickly without overwriting the pretrained features.
    The learning rate follows a cosine schedule over `epochs`.

    Args:
        model: Network to train, already on `device`.
        head: The model's new classifier layer.
        loss_fn: Loss taking (logits, targets).
        checkpoint: Path where the best weights and the full history are saved.
        config: Dict with 'epochs', 'lr_backbone', 'lr_head' and 'weight_decay'.

    Returns:
        A list with one dict of metrics per epoch. `model` ends up holding the best weights.
    """
    head_ids = {id(p) for p in head.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in head_ids], 'lr': config['lr_backbone']},
        {'params': head.parameters(), 'lr': config['lr_head']},
    ], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    history, best_acc, best_weights = [], -1.0, None
    for epoch in range(1, config['epochs'] + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        minutes = (time.time() - start) / 60
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, minutes=minutes))
        print(f'{checkpoint.stem} epoch {epoch}: train acc {train_acc:.3f} | val acc {val_acc:.3f} | {minutes:.1f} min')
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'weights': best_weights, 'history': history}, checkpoint)

    model.load_state_dict(best_weights)
    torch.save({'weights': best_weights, 'history': history}, checkpoint)
    return history


def train_or_load(name, loss_fn, train_loader, val_loader, num_classes, config, device, out_dir):
    """Trains an EfficientNet-B0 named `name`, or loads it if `out_dir/name.pt` already exists.

    Loading instead of retraining means the notebook can be re-run without hours of training;
    delete the checkpoint to retrain.

    Returns:
        (model, history DataFrame).
    """
    checkpoint = out_dir / f'{name}.pt'
    if checkpoint.exists():
        model, _, history = load_trained_efficientnet_b0(checkpoint, device)
        print(f'Loaded trained {name} from {out_dir.name}/{checkpoint.name}')
        return model, pd.DataFrame(history)

    model, head = build_efficientnet_b0(num_classes)
    model = model.to(device)
    history = fine_tune(model, head, loss_fn, train_loader, val_loader, checkpoint, config, device)
    return model, pd.DataFrame(history)
