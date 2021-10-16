import torch


def masked_cross_entropy(logits, target, mask):
    """
    logits: [batch, slots, len, vocab]
    target: [batch, slots, len]
    mask: [batch, slots, len]
    """

    logits_flat = logits.view(-1, logits.size(-1))  # [batch * slots * len, vocab]
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)  # [batch * slots * len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)  # [batch * slots * len, 1]
    losses = losses_flat.view(*target.size())
    loss = masking(losses, mask)
    
    return loss

def masking(losses, mask):
    """
    losses: [batch, slots, len]
    mask: [batch, slots, len]
    """

    losses = losses.masked_fill(mask, 0)
    loss = losses.sum() / (~mask).sum().float()

    return loss