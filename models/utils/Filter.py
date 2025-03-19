import cv2
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

class Filter:
    def __init__(self, device='cpu', num_gabor_filters=8, gabor_filter_kernel_size=5, log_filter_kernel_size=7):
        """
        Initialize the Filter class with device, number of Gabor filters, and filter kernel size.
        """
        self.device = device
        self.num_gabor_filters = num_gabor_filters
        self.gabor_filter_kernel_size = gabor_filter_kernel_size
        self.log_filter_kernel_size = log_filter_kernel_size

        # Get Gabor filters
        self.gabor_filters = self._create_gabor_filters()
        self.sobel_filter_x, self.sobel_filter_y = self._create_sobel_filters()
        self.log_filter_1, self.log_filter_2 = self._create_log_filter(self.log_filter_kernel_size)

    def _create_gabor_filters(self):
        """
        Create Gabor filters with different orientations.
        """
        filters = []
        for theta in np.linspace(0, np.pi, self.num_gabor_filters):
            kernel = cv2.getGaborKernel((self.gabor_filter_kernel_size, self.gabor_filter_kernel_size), 2.0, theta, 3.0, 0.5, 0, ktype=cv2.CV_32F)
            kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            filters.append(kernel_tensor)
        return nn.Parameter(torch.stack(filters).to(self.device), requires_grad=False)

    def _create_sobel_filters(self):
        """
        Create Sobel filters for edge detection in x and y directions.
        """
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return (torch.nn.Parameter(sobel_x.expand(1, 1, 3, 3).to(self.device), requires_grad=False),
                torch.nn.Parameter(sobel_y.expand(1, 1, 3, 3).to(self.device), requires_grad=False))

    def _create_log_filter(self, filter_kernel_size=7, sigma=1.0):
        """
        Create Laplacian of Gaussian (LoG) filter.
        """
        gaussian_filter = cv2.getGaussianKernel(filter_kernel_size, sigma)
        gaussian_filter = gaussian_filter @ gaussian_filter.T
        gaussian_filter_tensor = torch.tensor(gaussian_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        laplacian_kernel_1 = torch.tensor([[0.,  1., 0.],
                                           [1., -4., 1.],
                                           [0.,  1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                                           
        laplacian_kernel_2 = torch.tensor([[1.,  1., 1.],
                                           [1., -8., 1.],
                                           [1.,  1., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                                           
        log_filter_1 = F.conv2d(gaussian_filter_tensor.expand(1, 1, filter_kernel_size, filter_kernel_size), laplacian_kernel_1, padding=1)
        log_filter_2 = F.conv2d(gaussian_filter_tensor.expand(1, 1, filter_kernel_size, filter_kernel_size), laplacian_kernel_2, padding=1)
        
        log_filter_1 = log_filter_1.expand(1, 1, filter_kernel_size, filter_kernel_size).to(self.device)
        log_filter_2 = log_filter_2.expand(1, 1, filter_kernel_size, filter_kernel_size).to(self.device)
        
        return (nn.Parameter(log_filter_1, requires_grad=False), nn.Parameter(log_filter_2, requires_grad=False))

    def _grayscale_image(self, image):
        """
        Convert an RGB image to grayscale.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    def get_gabor_filtered_outputs(self, image, use_grayscale=False):
        """
        Apply Gabor filters to an RGB batch of images.
        Each RGB channel is filtered separately.
        """
        if use_grayscale:
            # Convert to grayscale (batch of images)
            if image.shape[1] == 3:  # Ensure it's RGB before conversion
                image = 0.2989 * image[:, 0:1, :, :] + 0.5870 * image[:, 1:2, :, :] + 0.1140 * image[:, 2:3, :, :]

        # Expand filters to match batch size
        B, C, H, W = image.shape
        # List to store outputs for each filter
        gabor_out = []
        for i in range(self.num_gabor_filters):
            # Apply Gabor filter to each channel independently
            filtered_image = F.conv2d(image, 
                                    self.gabor_filters[i].expand(C, 1, self.gabor_filter_kernel_size, self.gabor_filter_kernel_size).to(self.device), 
                                    padding=self.gabor_filter_kernel_size // 2, 
                                    groups=C)
            
            gabor_out.append(filtered_image)

        # Concatenate filtered outputs across the channel dimension
        #gabor_out = torch.cat(gabor_out, dim=1)

        #print(f'Gabor output shape: {gabor_out.shape}')  # Debugging

        return gabor_out
    
    def get_sobel_filtered_outputs(self, image):
        """
        Apply Sobel filters to a batch of images and return the filtered outputs.
        """
        # Convert to grayscale (batch of images)
        if image.shape[1] == 3:  # Ensure it's RGB before conversion
            image = 0.2989 * image[:, 0:1, :, :] + 0.5870 * image[:, 1:2, :, :] + 0.1140 * image[:, 2:3, :, :]

        # Expand filters to match batch size
        B, C, H, W = image.shape
        sobel_filter_x = self.sobel_filter_x.expand(C, 1, 3, 3)
        sobel_filter_y = self.sobel_filter_y.expand(C, 1, 3, 3)

        # Apply Sobel filters
        sobel_out_x = F.conv2d(image, sobel_filter_x, padding=1, groups=C)
        sobel_out_y = F.conv2d(image, sobel_filter_y, padding=1, groups=C)

        # Combine the outputs using the Sobel magnitude formula
        # sobel_out = torch.sqrt(sobel_out_x ** 2 + sobel_out_y ** 2)

        # # Normalize Sobel output while staying on GPU
        # sobel_out = (sobel_out - sobel_out.min()) / (sobel_out.max() - sobel_out.min())

        # # Convert to 3-channel by expanding (B, 1, H, W) → (B, 3, H, W)
        # sobel_rgb = sobel_out.expand(-1, 3, -1, -1)  # Expands across channel dimension

        return [sobel_out_x, sobel_out_y]  # This stays on GPU
    
    def get_log_filtered_outputs(self, image):
        """
        Apply Laplacian of Gaussian (LoG) filter to a batch of images and get the filtered outputs.
        """
        # Convert to grayscale (batch of images)
        if image.shape[1] == 3:  # Ensure it's RGB before conversion
            image = 0.2989 * image[:, 0:1, :, :] + 0.5870 * image[:, 1:2, :, :] + 0.1140 * image[:, 2:3, :, :]

        # Expand filters to match batch size
        B, C, H, W = image.shape
        log_filter_1 = self.log_filter_1.expand(C, 1, self.log_filter_kernel_size, self.log_filter_kernel_size).to(image.device)
        log_filter_2 = self.log_filter_2.expand(C, 1, self.log_filter_kernel_size, self.log_filter_kernel_size).to(image.device)

        # Apply LoG filters
        log_out_1 = F.conv2d(image, log_filter_1, padding=self.log_filter_kernel_size // 2, groups=C)
        log_out_2 = F.conv2d(image, log_filter_2, padding=self.log_filter_kernel_size // 2, groups=C)

        # Combine the outputs of the two LoG filters by taking the square root of the sum of their squares
        #log_out = torch.sqrt(log_out_1 ** 2 + log_out_2 ** 2)

        return [log_out_1, log_out_2]