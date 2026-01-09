import torch
import torch.nn as nn


def bbox_ciou(box1, box2, eps=1e-7):
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2

    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    rho2 = (box1[..., 0] - box2[..., 0]) ** 2 + \
           (box1[..., 1] - box2[..., 1]) ** 2

    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(box1[..., 2] / (box1[..., 3] + eps)) -
        torch.atan(box2[..., 2] / (box2[..., 3] + eps)), 2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - (rho2 / c2 + alpha * v)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.num_classes = num_classes

    def forward(self, preds, targets):
        preds = preds.permute(0, 2, 3, 1)

        pred_box = torch.sigmoid(preds[..., 0:4])
        pred_obj = preds[..., 4]
        pred_cls = preds[..., 5:]

        tgt_box = targets[..., 0:4]
        tgt_obj = targets[..., 4]
        tgt_cls = targets[..., 5:]

        obj_mask = tgt_obj == 1

        # -------- Box Loss --------
        if obj_mask.any():
            ciou = bbox_ciou(
                pred_box[obj_mask],
                tgt_box[obj_mask]
            )
            box_loss = (1 - ciou).mean()
        else:
            box_loss = torch.tensor(0.0, device=preds.device)

        # -------- Objectness Loss --------
        if obj_mask.any():
            obj_loss = self.bce(
                pred_obj[obj_mask],
                tgt_obj[obj_mask]
            ).mean()
        else:
            obj_loss = torch.tensor(0.0, device=preds.device)

        # -------- Classification Loss (BCE, one-vs-all) --------
        if obj_mask.any():
            obj_conf = torch.sigmoid(pred_obj[obj_mask]).detach()
            cls_raw = self.bce(
                pred_cls[obj_mask],
                tgt_cls[obj_mask]
            ).mean(dim=-1)
            cls_loss = (obj_conf * cls_raw).mean()
        else:
            cls_loss = torch.tensor(0.0, device=preds.device)

        return box_loss, obj_loss, cls_loss
