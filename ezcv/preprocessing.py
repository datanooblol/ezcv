import torchvision.transforms as transforms
img_size = 500
pipeline = transforms.Compose([
    transforms.ToPILImage(),          # Convert from OpenCV format (NumPy array) to PIL Image
    transforms.Resize((img_size, img_size)),   # Resize the image to 256x256 <- depend on the model
    transforms.RandomHorizontalFlip(p=1.0),  # Flip the image vertically (p=1.0 means always flip)
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
])