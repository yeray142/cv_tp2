from pprint import pprint
import os

import torchvision.transforms.functional

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms.functional as F
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes, save_image

from dataloader import get_loader
from models.yolo import YOLOv3
from utils.bbox import non_maximum_suppression, xywh2xyxy, get_map


def xywh2xyminmax(boxes):
    x, y, w, h = boxes
    return int(x - w / 2), int(x + w / 2), int(y - h / 2), int(y + h / 2)


def get_bbox(net: nn.Module, image: torch.Tensor, conf_threshold: float = 0.5):
    preds, anchors = net(image)
    bbox = {"class": [], "boxes": []}

    for pred, anchor in zip(preds, anchors):
        pred = pred.squeeze()

        for pred_bbox in pred:
            if pred_bbox[4] < conf_threshold:
                continue
            cls = pred_bbox[5:].argmax(0)

            bbox["boxes"] += [torch.tensor(xywh2xyminmax(pred_bbox[:4]))]
            bbox["class"] += [f"{cls}"]
    bbox["boxes"] = torch.stack(bbox["boxes"])
    return bbox


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def load_image(path: str):
    image = Image.open(path)
    x = F.to_tensor(image)
    x = x.unsqueeze(0)
    return x


if __name__ == "__main__":
    import torchvision.transforms.functional as TF
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    args = parser.parse_args()

    # logger = Logger("kilter-gallery", args, resume=True)
    # state_dict = logger.load_state_dict(best=True)

    num_classes = 1
    device = torch.device("cpu")
    net = YOLOv3(num_classes)
    net.load_state_dict(torch.load("checkpoints/exp6/best.pt", map_location="cpu")["state_dict"])
    net = net.to(device)
    net.eval()

    from albumentations.pytorch.transforms import ToTensorV2
    import albumentations as A

    val_transforms = A.Compose([
        A.LongestMaxSize(max_size=int(416)),
        A.PadIfNeeded(min_height=int(416), min_width=int(416),
                      border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        image = val_transforms(image=image)["image"].unsqueeze(0).to(device)

        preds = net(image)
        preds = non_maximum_suppression(preds)
        # res = get_map(preds, target)
        image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)[0]
        if preds is not None:
            image = draw_bounding_boxes(TF.resize(image, [416, 416]), xywh2xyxy(preds[0:1].cpu())[..., 0:4])
        image = np.transpose(image.cpu().numpy(), (1, 2, 0))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("asfd", image)
        cv2.waitKey(1)