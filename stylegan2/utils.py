def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad
