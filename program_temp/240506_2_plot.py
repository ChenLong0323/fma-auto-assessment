from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode

def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
):

    if img is None and isinstance(self.orig_img, torch.Tensor):
        img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

    names = self.names
    is_obb = self.obb is not None
    pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    annotator = Annotator(
        deepcopy(self.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Segment results
    if pred_masks and show_masks:
        if im_gpu is None:
            img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
            im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
            )
        idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {conf:.2f}" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

    # Plot Classify results
    if pred_probs is not None and show_probs:
        text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
        x = round(self.orig_shape[0] * 0.03)
        annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

    # Plot Pose results
    if self.keypoints is not None:
        for k in reversed(self.keypoints.data):
            annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

    # Show results
    if show:
        annotator.show(self.path)

    # Save results
    if save:
        annotator.save(filename)

    return annotator.result()