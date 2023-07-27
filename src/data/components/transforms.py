import torch
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from typing import Tuple, Optional, Union

def img_classification_transform(
    img,
    crop_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn"
):
    img = F.resize(img, resize_size, interpolation=interpolation, antialias=antialias)
    img = F.center_crop(img, crop_size)
    if not isinstance(img, torch.Tensor):
        img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=mean, std=std)
    
    return img