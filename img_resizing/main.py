from PIL import Image
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt

# Load the high-resolution image
original_image = Image.open('butterfly.png')

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(original_image)
plt.title('Original High-Resolution Image')
plt.axis('off')
plt.show()

# Convert to a smaller size (downscale), simulating a low-resolution image
low_res_image = original_image.resize((original_image.width // 4, original_image.height // 4), Image.BILINEAR)
plt.figure(figsize=(6, 6))
plt.imshow(low_res_image)
plt.title('Downscaled Low-Resolution Image')
plt.axis('off')
plt.show()

# Resize using Bilinear interpolation
bilinear_resized = low_res_image.resize(original_image.size, Image.BILINEAR)
plt.figure(figsize=(6, 6))
plt.imshow(bilinear_resized)
plt.title('Bilinear Interpolated Image')
plt.axis('off')
plt.show()

# Resize using Bicubic interpolation
bicubic_resized = low_res_image.resize(original_image.size, Image.BICUBIC)
plt.figure(figsize=(6, 6))
plt.imshow(bicubic_resized)
plt.title('Bicubic Interpolated Image')
plt.axis('off')
plt.show()

# Convert images to numpy arrays for numerical operations
original_array = np.array(original_image)
bilinear_array = np.array(bilinear_resized)
bicubic_array = np.array(bicubic_resized)

# Compute the Frobenius norm (L2 norm) for error estimation
bilinear_error = norm(original_array - bilinear_array)
bicubic_error = norm(original_array - bicubic_array)

print(f"Error (Frobenius norm) for Bilinear Interpolation: {bilinear_error}")
print(f"Error (Frobenius norm) for Bicubic Interpolation: {bicubic_error}")

# Experiment with different zoom factors (2x and 4x)
zoom_factors = [2, 4]
for zoom in zoom_factors:
    # Resize low-res image by different factors with bilinear and bicubic interpolation
    new_size = (original_image.width * zoom, original_image.height * zoom)

    bilinear_zoomed = low_res_image.resize(new_size, Image.BILINEAR)
    bicubic_zoomed = low_res_image.resize(new_size, Image.BICUBIC)

    # Display the zoomed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(bilinear_zoomed)
    plt.title(f'Bilinear Zoomed x{zoom}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(bicubic_zoomed)
    plt.title(f'Bicubic Zoomed x{zoom}')
    plt.axis('off')

    plt.show()
