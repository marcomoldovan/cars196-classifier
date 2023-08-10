import timm

avail_pretrained_models = timm.list_models(pretrained=True)
print(avail_pretrained_models)
