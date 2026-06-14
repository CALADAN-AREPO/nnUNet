from copy import deepcopy

import torch
import torch.nn.functional as F


class MeanTeacher:
    def __init__(self, student, ema_decay=0.999):
        self.teacher = deepcopy(student)
        self.ema_decay = ema_decay
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def update_teacher(self, student):
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), student.parameters()):
                tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * sp.data

    @torch.no_grad()
    def pseudo_label(self, unlabeled_batch, confidence_threshold=0.85):
        logits = self.teacher(unlabeled_batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        conf, pseudo = probs.max(dim=1)
        mask = conf > confidence_threshold
        return pseudo, mask

    def consistency_loss(self, student_logits, pseudo_labels, confidence_mask):
        if isinstance(student_logits, (list, tuple)):
            student_logits = student_logits[0]
        loss = F.cross_entropy(
            student_logits, pseudo_labels, reduction="none"
        )
        loss = loss * confidence_mask.float()
        return loss.mean()
