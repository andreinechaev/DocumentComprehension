import os
import gradio as gr
from PIL import Image
import cv2
import numpy as np
from scipy.ndimage import label
import torch
import torch.nn as nn
import torchvision.transforms as T


colors = {'table': (0, 0, 255), 'cell': (0, 255, 0)}


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(torch.cat([x2, x1], dim=1))
        out = self.output(x3)
        return out


# Load the model
unet_model = UNet(3, 2)
model_path = os.path.join(os.path.dirname(
    __file__), 'table_segmentation_model_quantized.pth')
print(f'Loading model from {model_path}')
unet_model.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu')))
unet_model.eval()
print('Model loaded successfully!', unet_model)


def extract_coordinates_from_mask(mask, threshold=0.5):
    # Apply the threshold to the mask
    binary_mask = (mask > threshold).astype(np.uint8)
    # Find connected components in the binary mask
    labeled_mask, num_components = label(binary_mask)
    coordinates = []
    for i in range(1, num_components + 1):
        component_mask = (labeled_mask == i)
        rows, cols = np.where(component_mask)

        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)

        coordinates.append({
            'label': 'table' if i == 1 else 'cell',
            'coordinates': {
                'x': int(x_min),
                'y': int(y_min),
                'width': int(x_max - x_min),
                'height': int(y_max - y_min)
            }
        })

    return coordinates


def predict_mask(model, image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    output_np = output.squeeze().detach().numpy()
    return np.argmax(output_np, axis=0)


def draw_rectangles(image, coordinates, colors):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for item in coordinates:
        label = item['label']
        coords = item['coordinates']
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        color = colors[label]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def predict_and_visualize(img):
    pil_img = Image.fromarray(np.uint8(img)).convert('RGB')
    mask = predict_mask(unet_model, pil_img)
    coordinates = extract_coordinates_from_mask(mask)
    marked_img = draw_rectangles(pil_img, coordinates, colors)

    return marked_img


input_image = gr.Image(shape=(256, 256))
output_image = gr.Image(type='pil', label='Detected Tables and Cells')

iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=input_image,
    outputs=output_image,
    title="Table and Cell Detection",
    description="Upload an image to detect tables and cells using the UNet model."
)

# Launch the Gradio interface
iface.launch(server_name='0.0.0.0', server_port=8094, debug=True)
