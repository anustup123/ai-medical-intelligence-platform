"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.

This is a self-contained implementation (no external grad-cam library
dependency) so the logic is fully transparent for the assignment write-up.

How it works:
1. Forward-hook captures the activations (feature maps) of the target
   convolutional layer.
2. Backward-hook captures the gradients of the class score w.r.t. those
   same activations.
3. Global-average-pool the gradients -> per-channel importance weights.
4. Weighted sum of the activation maps -> ReLU -> upsample to image size
   = the Grad-CAM heatmap.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: shape (1, 3, H, W), already normalized.
        class_idx: which class to explain. Defaults to the predicted class.
        Returns: heatmap (H, W) in range [0, 1], and the predicted class index + probs.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # (1, num_classes)
        probs = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]        # (C, h, w)
        activations = self.activations[0]    # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,) — global average pool
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        return cam, class_idx, probs.detach().cpu().numpy()[0]


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized (3, H, W) tensor back to an RGB uint8 image."""
    img = tensor.detach().clone().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(original_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays the Grad-CAM heatmap on top of the original image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap.astype(np.float32) * alpha +
               original_img.astype(np.float32) * (1 - alpha))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def describe_region(cam: np.ndarray) -> str:
    """
    Produces a short, human-readable textual description of WHERE the model's
    attention is concentrated (e.g. 'upper right lung field'). This text is
    fed into the LLM prompt so the generated report can refer to anatomical
    location without the LLM ever seeing the raw image.
    """
    h, w = cam.shape
    y, x = np.unravel_index(np.argmax(cam), cam.shape)

    vertical = "upper" if y < h / 3 else ("middle" if y < 2 * h / 3 else "lower")
    horizontal = "right" if x < w / 2 else "left"  # image-left = patient's right lung
    return f"{vertical} {horizontal} lung field"
