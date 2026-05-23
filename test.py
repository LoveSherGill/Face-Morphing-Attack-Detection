import os
import torch
from torchvision import models, transforms
from PIL import Image

# model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.eval()

image_extensions = (".jpg", ".jpeg", ".png")

image_file = None
for file in os.listdir():
    if file.endswith(image_extensions):
        image_file = file
        break

if image_file is None:
    print("No image found!")
    exit()

print("Using image:", image_file)

# load image
img = Image.open(image_file)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = transform(img).unsqueeze(0)

# prediction
output = model(img)
_, pred = torch.max(output, 1)

if pred.item() == 0:
    print("REAL FACE")
else:
    print("MORPHED FACE")