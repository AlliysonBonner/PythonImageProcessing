from image import Image
import numpy as np

def brighten(image: Image, factor: float) -> Image:
    """
    Brightens the input image by multiplying each channel with the given factor.

    Args:
        image: An instance of the Image class representing the input image.
        factor: A float value representing the amount to brighten the image by.

    Returns:
        An instance of the Image class representing the brightened image.
    """
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    # Vectorized version that leverages NumPy for faster computation.
    new_im.array = image.array * factor
    return new_im

def adjust_contrast(image: Image, factor: float, mid: float) -> Image:
    """
    Adjusts the contrast of the input image by increasing the difference from the user-defined midpoint by factor amount.

    Args:
        image: An instance of the Image class representing the input image.
        factor: A float value representing the amount to increase the difference from the midpoint by.
        mid: A float value representing the user-defined midpoint.

    Returns:
        An instance of the Image class representing the contrast-adjusted image.
    """
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    for x in range(image.x_pixels):
        for y in range(image.y_pixels):
            for c in range(image.num_channels):
                new_im.array[x, y, c] = (image.array[x, y, c] - mid) * factor + mid
    return new_im

def blur(image: Image, kernel_size: int) -> Image:
    """
    Applies a blur effect on the input image by taking into account a number of neighboring pixels determined by the given kernel size.

    Args:
        image: An instance of the Image class representing the input image.
        kernel_size: An integer value representing the number of pixels to take into account when applying the blur.

    Returns:
        An instance of the Image class representing the blurred image.
    """
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    neighbor_range = kernel_size // 2
    for x in range(image.x_pixels):
        for y in range(image.y_pixels):
            for c in range(image.num_channels):
                total = 0
                for x_i in range(max(0, x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0, y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size ** 2)
    return new_im

def apply_kernel(image: Image, kernel: np.ndarray) -> Image:
    """Applies a 2D convolution kernel to an image.

    Args:
        image (Image): The image to apply the kernel to.
        kernel (np.ndarray): The 2D convolution kernel.

    Returns:
        Image: The image after the kernel has been applied.
    """
    x_pixels, y_pixels, num_channels = image.array.shape
    # Create a new Image object to store the output
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    # Calculate the range of neighboring pixels to consider based on the kernel size
    neighbor_range = kernel.shape[0] // 2

    # Iterate over each pixel in the image
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                # Iterate over the neighboring pixels within the range defined by the kernel size
                for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        # Apply the kernel to the pixel and add to the total
                        total += image.array[x_i, y_i, c] * kernel_val
                # Store the result in the output image
                new_im.array[x, y, c] = total

    return new_im

def combine_images(image1: Image, image2: Image) -> Image:
    """Combines two images by taking the squared sum of squares.

    Args:
        image1 (Image): The first image.
        image2 (Image): The second image.

    Returns:
        Image: The combined image.
    """
    assert image1.array.shape == image2.array.shape, "Images must be the same size."

    x_pixels, y_pixels, num_channels = image1.array.shape
    # Create a new Image object to store the output
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    # Iterate over each pixel in the images
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                # Calculate the squared sum of squares and take the square root
                new_im.array[x, y, c] = (image1.array[x, y, c]**2 + image2.array[x, y, c]**2)**0.5

    return new_im

if __name__ == '__main__':
    cake = Image(filename='cake.png')
    city = Image(filename='city.png')

    # brightening
    brightened_im = brighten(cake, 1.7)
    brightened_im.write_image('brightened.png')

    # darkening
    darkened_im = brighten(cake, 0.3)
    darkened_im.write_image('darkened.png')

    # increase contrast
    incr_contrast = adjust_contrast(cake, 2, 0.5)
    incr_contrast.write_image('increased_contrast.png')

    # decrease contrast
    decr_contrast = adjust_contrast(cake, 0.5, 0.5)
    decr_contrast.write_image('decreased_contrast.png')

    # blur using kernel 3
    blur_3 = blur(city, 3)
    blur_3.write_image('blur_k3.png')

    # blur using kernel size of 15
    blur_15 = blur(city, 15)
    blur_15.write_image('blur_k15.png')

    # let's apply a sobel edge detection kernel on the x and y axis
    sobel_x = apply_kernel(city, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    sobel_x.write_image('edge_x.png')
    sobel_y = apply_kernel(city, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_y.write_image('edge_y.png')

    # let's combine these and make an edge detector!
    sobel_xy = combine_images(sobel_x, sobel_y)
    sobel_xy.write_image('edge_xy.png')