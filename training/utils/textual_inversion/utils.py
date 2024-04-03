def freeze_params(params):
    for param in params:
        param.requires_grad = False