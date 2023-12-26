import torch
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

def load_model(model_path):
    """Load the pre-trained ESRGAN model"""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    return model

def upscale_image(image_path, model):
    """Upscale an image using the pre-trained ESRGAN model"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output

# Example usage
model_path = './2xLexicaRRDBNet_Sharp.pth'  # You need to download the pre-trained model
image_path = 'images/face_age/024/024_6422.png'

model = load_model(model_path)
upscaled_image = upscale_image(image_path, model)
cv2.imwrite('upscaled_image.jpg', upscaled_image)
