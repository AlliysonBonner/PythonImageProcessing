
**Oirginal image**
![a collage of proccessed images from the project](https://github.com/AlliysonBonner/PythonImageProcessing/blob/main/image.png?raw=true)
This project demonstrates how to perform various image manipulations using Python. The project includes a Python module transform.py which provides the following functions:
**Oirginal image**

* `brighten(image, factor): brightens the input image by the given factor.
* adjust_contrast(image, factor, mid): adjusts the contrast of the input image by increasing or decreasing the difference from the given midpoint by the given factor.
* blur(image, kernel_size): applies a blur effect to the input image using the given kernel size.
* apply_kernel(image, kernel): applies the given kernel to the input image.
* combine_images(image1, image2): combines two images using the squared sum of squares.
The transform.py module uses the Image class from the image.py module to read and write image files. The Image class provides methods to load and save images in various formats.
## Requirements
This project requires Python 3 and the following Python packages:
* numpy
* imageio
## Installation
To install the required packages, run the following command:
`pip install -r requirements.txt`
## Usage
To use the transform.py module, import the required functions and create an instance of the Image class with the input image file:
```
from image import Image
from transform import brighten, adjust_contrast, blur, apply_kernel, combine_images

image = Image(filename='input.png')

brightened_im = brighten(image, 1.7)
brightened_im.write_image('brightened.png')

incr_contrast = adjust_contrast(image, 2, 0.5)
incr_contrast.write_image('increased_contrast.png')

blur_3 = blur(image, 3)
blur_3.write_image('blur_k3.png')

sobel_x = apply_kernel(image, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
sobel_y = apply_kernel(image, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
sobel_xy = combine_images(sobel_x, sobel_y)
sobel_xy.write_image('edge_xy.png')
```
The above code loads an image from a file named input.png and performs various transformations on it using the transform.py module. The resulting images are saved to files in the current directory.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
