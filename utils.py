import cv2
import numpy as np
import torch
image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


def image_normalize(image, size=224):
    image_array = cv2.imread(image)
    image_array = cv2.resize(image_array, (size, size))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
    image_array = image_array.transpose((2, 0, 1))
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor


def input_dim_3to4(tensor):
    img_tensor_list = [torch.unsqueeze(tensor, 0)]
    img_tensor = torch.cat(tuple(img_tensor_list), dim=0)
    return img_tensor
