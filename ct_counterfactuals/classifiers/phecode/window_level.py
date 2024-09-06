import torch

abd_windows = dict(
    soft_tissue = (50, 400),
    liver_cect = (80, 150),
    bone = (400, 1800)
)

def apply_window_level(img, windows=abd_windows):
    """
    Apply window leveling to an image with specified windows for each channel.
    
    Parameters:
    img (torch.Tensor): The input image tensor.
    windows (dict): A dictionary containing window settings for each channel.
    
    Returns:
    torch.Tensor: The window-leveled image.
    """
    # Initialize an empty tensor to hold the window-leveled image
    leveled_img = torch.zeros_like(img)
    
    # Iterate over each channel and apply the window level
    for channel, (WL, WW) in enumerate(windows.values()):
        # Calculate the min and max window values
        min_val = WL - (WW / 2)
        max_val = WL + (WW / 2)
        
        # Apply window leveling: Scale the intensity values to be within the window
        # Values below min_val are set to 0, and values above max_val are set to 1
        # Values within the window are scaled linearly between 0 and 1
        leveled_img[:, channel, ...] = torch.clamp((img[:, channel, ...] - min_val) / (max_val - min_val), 0, 1)
    
    return leveled_img
    