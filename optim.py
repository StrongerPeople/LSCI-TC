from torch.optim import AdamW

def create_optimizer(args, model):
    lr = args.lr
    wd = args.weight_decay
    lr_mult = getattr(args, 'lr_mult', 1)     
    decay_lr_mult = getattr(args, 'decay_lr_mult', 1)
    
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.05, "lr": lr},  
        {"params": [], "weight_decay": 0.05, "lr": lr * lr_mult}, 
        {"params": [], "weight_decay": 0.05, "lr": lr * decay_lr_mult},  
        {"params": [], "weight_decay": wd, "lr": lr},  
        {"params": [], "weight_decay": wd, "lr": lr * lr_mult}, 
        {"params": [], "weight_decay": wd, "lr": lr * decay_lr_mult},
    ]

    decay_keywords = {""}
    # decay_keywords = {""}
    vision_keywords = {""} 
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  
        need_decay = any(kw in n for kw in decay_keywords)
        if any(kw in n for kw in vision_keywords):
            group_offset = 1  
        elif any(module in n for module in decay_keywords):
            group_offset = 2  
        else:
            group_offset = 0  
        group_idx = group_offset + (3 if need_decay else 0)
        optimizer_grouped_parameters[group_idx]['params'].append(p)

    return AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    
