import torch
import matplotlib.pyplot as plt

def show_image_comparison(pil_img, tensor_img, mean=None, std=None):
    """
    Display raw PIL image vs. tensor image (optionally unnormalized).
    
    pil_img: PIL image (original)
    tensor_img: torch.Tensor (C,H,W) already transformed
    mean, std: normalization values (list of floats) if used
    """
    # Make sure tensor is detached and on CPU
    img_tensor = tensor_img.detach().cpu()

    if mean is not None and std is not None:
        mean = torch.tensor(mean)[:, None, None]
        std = torch.tensor(std)[:, None, None]
        img_tensor = img_tensor * std + mean

    # Clip to [0,1] for safe display
    img_tensor = img_tensor.clamp(0, 1)

    # Convert to HWC
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pil_img)
    axes[0].set_title("Original PIL")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].set_title("Network Input (unnormalized)")
    axes[1].axis("off")

    plt.show()