from __future__ import annotations
import cv2, numpy as np
from typing import Tuple

# Prøv at indlæse torch/torchvision
_TORCH_OK = False
try:
    import torch  # type: ignore
    import torchvision  # type: ignore
    from torchvision.transforms.functional import to_tensor  # type: ignore
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


class SilhouetteSegmenter:
    """
    Person-silhuet som uint8 maske (0/255).
    Default: DeepLabV3 (torchvision). Fallback: OpenCV HOG+GrabCut.
    """
    def __init__(self, threshold: float = 0.5, prefer_backend: str = "deeplabv3", device: str = "cpu"):
        self.threshold = float(threshold)
        self.prefer_backend = prefer_backend.lower()
        self.device = device if device in ("cpu", "cuda") else "cpu"
        self.backend = "opencv"
        self.model = None
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.person_idx = 15

        if self.prefer_backend in ("deeplabv3", "auto"):
            if _TORCH_OK:
                try:
                    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                    m = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
                    self.model = m.to(self.device).eval()
                    self.backend = "deeplabv3"
                except Exception:
                    self.model = None
                    if self.prefer_backend == "deeplabv3":
                        raise RuntimeError("DeepLabV3 kunne ikke initialiseres. Tjek torch/torchvision.")
            elif self.prefer_backend == "deeplabv3":
                raise RuntimeError("PyTorch/torchvision ikke fundet; kan ikke køre DeepLabV3.")

    def _deeplab_mask(self, bgr: np.ndarray) -> np.ndarray:
        import torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = to_tensor(rgb).unsqueeze(0)
        mean = torch.tensor(self.mean, device=self.device)[:, None, None]
        std  = torch.tensor(self.std,  device=self.device)[:, None, None]
        x = (x.to(self.device) - mean) / std
        with torch.inference_mode():
            out = self.model(x)["out"][0]      # [C,H,W]
            probs = out.softmax(0)
            person = probs[self.person_idx].detach().cpu().numpy()
        mask = (person > self.threshold).astype(np.uint8) * 255
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return mask

    def _opencv_mask(self, bgr: np.ndarray) -> np.ndarray:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, _ = hog.detectMultiScale(bgr, winStride=(8,8), padding=(8,8), scale=1.05)
        if len(rects) > 0:
            x, y, w, h = max(rects, key=lambda r: r[2]*r[3])
        else:
            H, W = bgr.shape[:2]
            w, h = int(W*0.5), int(H*0.8)
            x, y = (W-w)//2, int(H*0.1)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(bgr, mask, (x,y,w,h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask_bin = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k = np.ones((5,5), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k)
        return mask_bin

    def __call__(self, bgr: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.backend == "deeplabv3" and self.model is not None:
            return self._deeplab_mask(bgr), "deeplabv3"
        return self._opencv_mask(bgr), "opencv"
