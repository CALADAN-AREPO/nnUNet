import torch
import torch.nn.functional as F
from torch.optim import Adam


def normalize_per_image(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def tent_adapt(model, test_batch, steps=1, lr=1e-4):
    bn_params = []
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            for p in mod.parameters():
                p.requires_grad = True
                bn_params.append(p)
            mod.train()
        else:
            for p in mod.parameters(recurse=False):
                p.requires_grad = False

    optimizer = Adam(bn_params, lr=lr)
    for _ in range(steps):
        logits = model(test_batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)
        loss = -(probs * torch.log(probs.clip(min=1e-8))).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def flip_tta(model, x):
    with torch.no_grad():
        pred_orig = model(x)
        if isinstance(pred_orig, (list, tuple)):
            pred_orig = pred_orig[0]
        pred_flip = model(x.flip(-1))
        if isinstance(pred_flip, (list, tuple)):
            pred_flip = pred_flip[0]
        pred_flip = pred_flip.flip(-1)
    return 0.5 * (pred_orig + pred_flip)


def gamma_tta(model, x, gammas=(0.85, 1.0, 1.15)):
    preds = []
    with torch.no_grad():
        for g in gammas:
            x_g = x ** g
            pred = model(x_g)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            preds.append(pred)
    return sum(preds) / len(preds)
