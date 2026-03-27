import torch

def boxes_to_mask(batch_idx, bboxes_xywhn, image_shape, batch_size, device):
    h, w = image_shape
    mask = torch.zeros((batch_size, 1, h, w), device=device, dtype=torch.float32)
    if bboxes_xywhn.numel() == 0:
        return mask
    for i in range(bboxes_xywhn.shape[0]):
        bi = int(batch_idx[i].item())
        x, y, bw, bh = bboxes_xywhn[i]
        x1 = int(max(0, (x - bw / 2) * w))
        y1 = int(max(0, (y - bh / 2) * h))
        x2 = int(min(w, (x + bw / 2) * w))
        y2 = int(min(h, (y + bh / 2) * h))
        if x2 > x1 and y2 > y1:
            mask[bi, :, y1:y2, x1:x2] = 1.0
    return mask
