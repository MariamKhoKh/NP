import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity as ssim


def read_image(path):
    '''Read image and return the image properties.
    Parameters:
    path (string): Image path

    Returns:
    numpy.ndarray: Image exists in "path"
    tuple: Image dimension (number of rows and columns)
    '''
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"The image at {path} could not be found.")

    dimension = (img.shape[0], img.shape[1])

    return img, dimension


def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    '''Resize image to a specific scale of original image.
    Parameters:
    img (numpy.ndarray): Original image
    dimension (tuple): Original image dimension
    scale (int): Multiply the size of the original image

    Returns:
    numpy.ndarray: Resized image
    '''
    scale /= 100
    new_dimension = (int(dimension[1] * scale), int(dimension[0] * scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)

    return resized_img


def nearest_interpolation(image, dimension):
    '''Optimized nearest neighbor interpolation using numpy indexing.'''
    scale_x = dimension[1] / image.shape[1]
    scale_y = dimension[0] / image.shape[0]
    row_indices = np.clip((np.arange(dimension[0]) / scale_y).astype(int), 0, image.shape[0] - 1)
    col_indices = np.clip((np.arange(dimension[1]) / scale_x).astype(int), 0, image.shape[1] - 1)
    new_image = image[row_indices[:, None], col_indices]
    return new_image


def bilinear_interpolation(image, dimension):
    '''Optimized bilinear interpolation using numpy operations.'''
    height, width = image.shape[:2]
    new_height, new_width = dimension
    scale_x = width / new_width
    scale_y = height / new_height
    x_coords = (np.arange(new_width) + 0.5) * scale_x - 0.5
    y_coords = (np.arange(new_height) + 0.5) * scale_y - 0.5
    x_coords = np.clip(x_coords, 0, width - 2)
    y_coords = np.clip(y_coords, 0, height - 2)
    x0 = x_coords.astype(int)
    y0 = y_coords.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    x_diff = x_coords - x0
    y_diff = y_coords - y0
    new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    for c in range(3):
        Ia = image[y0[:, None], x0, c]
        Ib = image[y0[:, None], x1, c]
        Ic = image[y1[:, None], x0, c]
        Id = image[y1[:, None], x1, c]
        new_image[:, :, c] = (
            Ia * (1 - x_diff) * (1 - y_diff[:, None]) +
            Ib * x_diff * (1 - y_diff[:, None]) +
            Ic * (1 - x_diff) * y_diff[:, None] +
            Id * x_diff * y_diff[:, None]
        )
    return np.clip(new_image, 0, 255).astype(np.uint8)


def bicubic_interpolation(image, dimension):
    '''Optimized bicubic interpolation using scipy’s map_coordinates.'''
    height, width = image.shape[:2]
    new_height, new_width = dimension
    scale_x = width / new_width
    scale_y = height / new_height
    x_coords = (np.arange(new_width) + 0.5) * scale_x - 0.5
    y_coords = (np.arange(new_height) + 0.5) * scale_y - 0.5
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    output = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    for c in range(3):
        output[:, :, c] = map_coordinates(
            image[:, :, c],
            [y_grid.ravel(), x_grid.ravel()],
            order=3,
            mode='reflect'
        ).reshape((new_height, new_width))
    return np.clip(output, 0, 255).astype(np.uint8)


def mse_error(img1, img2):
    '''Calculate the Mean Squared Error between two images'''
    error = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    return error


def ssim_error(img1, img2):
    '''Calculate SSIM between two images'''
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return 1 - score  # Lower score indicates less similarity


def show_result(images_list):
    '''Display the images'''
    titles = list(images_list.keys())
    images = list(images_list.values())
    rows = len(images_list) // 3 + 1
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    fig.suptitle('Image Interpolation Comparison', fontsize=16)
    for idx, (title, img) in enumerate(images_list.items()):
        ax = axs[idx // 3, idx % 3]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.show()


def main(image_path):
    images_list = {}
    img, dimension = read_image(image_path)
    images_list['Original Image'] = img
    scale_percent = 31
    resized_img = image_change_scale(img, dimension, scale_percent)
    images_list['Smaller Image'] = resized_img
    nn_img_algo = nearest_interpolation(resized_img, dimension)
    images_list['Nearest Neighbor Interpolation'] = nn_img_algo
    bil_img_algo = bilinear_interpolation(resized_img, dimension)
    images_list['Bilinear Interpolation'] = bil_img_algo
    cubic_img_algo = bicubic_interpolation(resized_img, dimension)
    images_list['Bicubic Interpolation'] = cubic_img_algo
    error_list_mse = [
        mse_error(nn_img_algo, img),
        mse_error(bil_img_algo, img),
        mse_error(cubic_img_algo, img)
    ]
    error_list_ssim = [
        ssim_error(nn_img_algo, img),
        ssim_error(bil_img_algo, img),
        ssim_error(cubic_img_algo, img)
    ]
    print("MSE Errors:", error_list_mse)
    print("SSIM Errors:", error_list_ssim)
    show_result(images_list)
    interpolation_methods = ["Nearest Neighbor", "Bilinear", "Bicubic"]
    plt.bar(interpolation_methods, error_list_mse, color=['blue', 'green', 'purple'])
    plt.title("MSE Error Comparison")
    plt.ylabel("MSE")
    plt.show()
    plt.bar(interpolation_methods, error_list_ssim, color=['blue', 'green', 'purple'])
    plt.title("SSIM Error Comparison")
    plt.ylabel("1 - SSIM")
    plt.show()


main("file.png")
