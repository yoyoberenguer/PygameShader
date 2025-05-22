Shader_gpu
========================================


:mod:`Shader_gpu.pyx`

---------------------

.. _gpu_image_processing_summary:

1. GPU-Accelerated Image Processing Library
-------------------------------------------

This library provides GPU-accelerated image processing functionality
leveraging **CUDA** (via **CuPy**) to efficiently perform various image
transformations and filters on **NVIDIA GPUs**. Its main purpose is to
speed up computationally expensive image processing operations that would
otherwise be slow on a CPU.

2. Key Features & Use Cases
---------------------------

**GPU Memory & Hardware Information**
Retrieve details about the available GPU memory, PCI bus ID, and compute capability.
**Use Case**: Useful for dynamically managing memory allocation and optimizing parallel processing tasks.

**Image Processing & Filters**
 Perform common image filters efficiently on the GPU, including:

- **Inversion**: :func:`invert_gpu`
- **Grayscale & Sepia**: :func:`grayscale_gpu`, :func:`sepia_gpu`
- **Edge Detection**: Sobel, Prewitt, Canny
- **Blur & Smoothing Filters**: Gaussian, Median, Bilateral
- **Embossing & Sharpening**

3. Use Case
-----------

 Optimized for real-time video and image processing by executing these operations in parallel on the GPU.

**Color Manipulation & Enhancements**
Adjust image properties such as brightness, contrast, saturation, and HSL/HSV values.

**Use Case**: Ideal for image enhancement, augmented reality (AR) applications, and photo editing software.

**Special Effects & Transformations**
Apply artistic transformations and effects, including:

- **Swirl, Fisheye, Wave, Ripple Effects**
- **Chromatic Aberration & RGB Splitting**
- **Cartoon Effect & Bloom Filters**
- **Dithering & Heatmap Effects**

**Use Case**: Useful for game development, VFX, and computer graphics applications.

**Geometric Transformations**
Support for geometric transformations such as:

- **Mirroring**
- **Zooming**
- **Downscaling**
- **Blending**

**Use Case**: Can be used for image compression, texture mapping, and real-time rendering.

**Real-Time Video Processing**
Real-time video effects are supported by functions such as :func:`ripple_effect_gpu()`,
:func:`predator_gpu()`, and :func:`heatmap_gpu()`.

**Use Case**: Ideal for integration into streaming software, video filters, or surveillance systems.


4. Potential Applications
-------------------------

- **Real-time Image & Video Processing**: Enhance camera feeds in real-time.
- **Computer Vision & AI Preprocessing**: Apply fast filters before passing images to ML models.
- **Game Development & Graphics**: Add special effects for gaming & simulation environments.
- **Augmented & Virtual Reality (AR/VR)**: Optimize visuals for immersive experiences.
- **High-performance Photography Tools**: Edit large images efficiently using the GPU.


5. Summary
----------

This library offloads complex image processing tasks from the CPU to the GPU,
making operations significantly faster and more efficient. It is ideal for applications
that require real-time performance, such as games, AR/VR, video streaming, and AI-based
image processing.

6. Cython list methods
----------------------

|

.. currentmodule:: Shader_gpu

|

.. py:function:: get_gpu_free_mem()

   Retrieve the available free memory on the GPU.

   This function queries the GPU device for its current free memory and
   returns the amount in bytes. It is useful for monitoring memory usage
   when working with GPU-accelerated computations, ensuring that operations
   do not exceed available memory.

   **Returns:**

   - **int**: The amount of free GPU memory in bytes.

   **Example Usage:**

   .. code-block:: python

      free_mem = get_gpu_free_mem()
      print(f"Available GPU Memory: {free_mem / (1024**2):.2f} MB")

|

.. py:function:: get_gpu_maxmem()

   Retrieve the total memory capacity of the GPU.

   This function queries the GPU device to determine its total available
   memory, representing the total VRAM capacity of the GPU.

   **Returns:**

   - **int**: The total GPU memory in bytes.

   **Example Usage:**

   .. code-block:: python

      total_mem = get_gpu_maxmem()
      print(f"Total GPU Memory: {total_mem / (1024**2):.2f} MB")

|

.. py:function:: get_gpu_pci_bus_id()

   Retrieve the PCI Bus ID of the GPU.

   This function queries the GPU device and returns its PCI Bus ID,
   which uniquely identifies the GPU within the system. The PCI Bus ID
   is useful for multi-GPU setups, debugging, and device management.

   **Returns:**

   - **str**: The PCI Bus ID of the GPU in the format "Domain:Bus:Device.Function"
     (e.g., "0000:01:00.0").

   **Example Usage:**

   .. code-block:: python

      pci_id = get_gpu_pci_bus_id()
      print(f"GPU PCI Bus ID: {pci_id}")

|

.. py:function:: get_compute_capability()

   Retrieve the compute capability of the GPU.

   Compute capability represents the GPU architecture version and determines
   its compatibility with various CUDA features. Higher compute capability
   values indicate support for more advanced features and optimizations.

   **Returns:**

   - **str**: The compute capability of the GPU as a string in the format "major.minor"
     (e.g., "7.5" for NVIDIA Turing GPUs).

   **Example Usage:**

   .. code-block:: python

      capability = get_compute_capability()
      print(f"GPU Compute Capability: {capability}")

|

.. py:function:: get_max_grid_per_block()

   Retrieve the maximum number of grid blocks per multiprocessor.

   This function returns the maximum number of grid blocks that can be active
   per multiprocessor on the GPU. This value is essential for optimizing
   parallel workloads and ensuring efficient resource utilization.

   **Returns:**

   - **unsigned int**: The maximum number of grid blocks per multiprocessor.

   **Example Usage:**

   .. code-block:: python

      max_grid_blocks = get_max_grid_per_block()
      print(f"Max grid blocks per multiprocessor: {max_grid_blocks}")

   **Notes:**

   - The returned value depends on the GPU's compute capability.
   - For instance, GPUs with Compute Capability 2.x support up to
     8 active blocks per multiprocessor, while Compute Capability 3.x
     and higher can support up to 16 or more active blocks per multiprocessor.
     Refer to NVIDIA's CUDA programming guide for specific values per architecture.

|

.. py:function:: block_grid(w, h)

   Automatically computes optimal grid and block sizes for GPU execution.

   This function determines the best grid and block configuration for GPU execution based on
   the dimensions of the display (or computational domain). It ensures that the computed grid
   and block sizes are valid and compatible with the given width (`w`) and height (`h`).

   **Parameters:**

   - **w** (*int*): The width of the display or computational domain. Must be greater than 0.
   - **h** (*int*): The height of the display or computational domain. Must be greater than 0.

   **Returns:**

   - **tuple**: A tuple containing:
     - `grid`: (*y, x*) - The computed grid size (number of blocks in each dimension).
     - `block`: (*yy, xx*) - The computed block size (number of threads per block in each dimension).

   **Raises:**

   - **AssertionError**:
     - If `w` or `h` is less than or equal to 0.
     - If the computed grid and block sizes are not valid (i.e., they do not exactly cover the input dimensions).

   **Notes:**

   - The function first determines possible divisors of `w` and `h`, then selects block sizes that
     do not exceed 32 (a common maximum block size for GPU execution).
   - The final grid size is determined by dividing the input dimensions by the selected block sizes.
   - If the computed configuration is invalid (i.e., `yy * y != h` or `xx * x != w`), an assertion
     error is raised, suggesting that manual configuration might be necessary.

   **Example:**

   .. code-block:: python

      grid, block = block_grid(128, 64)
      print(grid, block)  # Output: (4, 4) (16, 16)

|

.. py:function:: conv(v)

   Convert a value to a human-readable format,
   scaling it by powers of 1024 (e.g., KB, MB, GB).

   This function divides the input value `v` by 1024 repeatedly until
   it is less than or equal to 1024. It then returns the value with its
   appropriate unit (e.g., KB, MB, GB) based on the number of divisions performed.

   **Parameters:**

   - **v** (*float* or *int*): The value to be converted (typically representing bytes or data size).

   **Returns:**

   - **str**: A human-readable string representing the value with an appropriate unit, rounded to 3 decimal places.

   **Notes:**

   - The function uses the following units:
     - 1024 bytes = 1 KB
     - 1024 KB = 1 MB
     - 1024 MB = 1 GB
     - and so on...

   - If the input value `v` is smaller than 1024, it will be returned as is with the first unit.
   - The function ensures that the appropriate unit is selected based on the scale of `v`.

   **Example:**

   .. code-block:: python

      conv(1048576)  # Returns '1.0 MB'
      conv(123456789)  # Returns '117.74 MB'

|

.. py:function:: block_and_grid_info(w, h)

   Calculate and display the appropriate GPU block and grid dimensions for a given image size.

   This function computes the grid and block sizes required for optimal GPU parallelization.
   It utilizes the `block_grid` function to determine the best block and grid sizes based on the
   provided width (`w`) and height (`h`) of the image or display.

   The block and grid sizes are then validated to ensure they can correctly cover the entire area.
   If the validation fails, assertions are raised. Afterward, the function prints the resulting
   grid and block sizes.

   **Parameters:**

   - **w** (*int*): The width of the image or display (must be greater than 0).
   - **h** (*int*): The height of the image or display (must be greater than 0).

   **Raises:**

   - **AssertionError**:
     - If either `w` or `h` is non-positive.
     - If the calculated grid and block sizes do not properly cover the image.

   **Returns:**

   - **None**: This function only prints the GPU grid and block information. It does not return any values.

   **Example:**

   .. code-block:: python

      block_and_grid_info(1920, 1080)
      # Output:
      # GPU GRID        : (grid_y=34.000000, grid_x=60.000000)
      # GPU BLOCK       : (block_y=32.000000, block_x=32.000000)

   **Notes:**

   - The grid is determined by the number of blocks required to cover the image's height and width.
   - The block size refers to the size of each individual block of threads in the GPU.
   - This function assumes the GPU processing will be optimized for the calculated grid and block sizes.

|

.. py:function:: get_gpu_info()

   Display detailed GPU information, including memory, compute capability, and PCI bus ID.

   This function retrieves and prints various details about the GPU, including:
   - The installed CuPy version.
   - The maximum grid size per block supported by the GPU.
   - The available and total GPU memory, both in bytes and human-readable formats.
   - The PCI Bus ID of the GPU.
   - The compute capability of the GPU.

   **Returns:**

   - **None**: This function prints GPU-related information to the console.

   **Example output:**

   .. code-block:: python

      get_gpu_info()
      # Output:
      # CUPY VERSION           : 11.2.0
      # GPU MAX GRID PER BLOCK : 65535
      # GPU FREE MEMORY        : (mem=8388608000.000000, (8.0 GB))
      # GPU MAX MEMORY         : (mem=17179869184.000000, (16.0 GB))
      # GPU PCI BUS ID         : (bus=0000:01:00.0)
      # GPU CAPABILITY         : (capa=7.5)

   **Notes:**

   - The function relies on CuPy for GPU memory information.
   - The `conv` function is used to convert raw memory values into a more readable format (e.g., GB).
   - This function is useful for debugging and verifying GPU resource availability before launching CUDA-based operations.

|

.. py:function:: invert_gpu(surface_)

   Perform GPU-accelerated color inversion on a Pygame surface (returns a copy).

   This function transfers the image data from a `pygame.Surface` to the GPU,
   inverts the color values using CuPy, and then reconstructs a new surface
   with the modified pixel data.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A Pygame surface object (24-bit or 32-bit) containing the image to be inverted.

   **Returns:**

   - **pygame.Surface**: A new Pygame surface with inverted colors, preserving the original format (BGR or BGRA).

   **Raises:**

   - **ValueError**: If the surface cannot be referenced as a 1D buffer or has zero length.
   - **TypeError**: If `surface_` is not a `pygame.Surface` object.

   **Notes:**

   - Uses CuPy to perform the inversion directly on the GPU for optimal performance.
   - Automatically detects whether the surface is 24-bit (BGR) or 32-bit (BGRA).
   - Synchronizes the GPU to ensure completion before returning the processed image.

   **Example:**

   .. code-block:: python

      import pygame
      surface = pygame.image.load("image.png").convert()
      inverted_surface = invert_gpu(surface)
      pygame.image.save(inverted_surface, "inverted_image.png")

|

.. py:function:: invert_gpu_inplace(surface_)

   Perform in-place GPU-accelerated color inversion on a Pygame surface.

   This function directly modifies the pixel data of the given `pygame.Surface`
   by inverting its color values using GPU processing via CuPy. The inversion
   is performed in-place, meaning the original surface is altered without
   creating a new one.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The Pygame surface whose pixel colors will be inverted.
     Must be a valid 24-bit (BGR) or 32-bit (BGRA) surface.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface` object.
   - **ValueError**: If the surface buffer cannot be accessed or has zero length.

   **Notes:**

   - The function transfers the pixel data to the GPU, performs the inversion,
     and copies the modified pixels back to the original surface.
   - Uses CuPy for efficient GPU computation.

   **Example:**

   .. code-block:: python

      import pygame

      pygame.init()
      surface = pygame.image.load("image.png").convert()
      invert_gpu_inplace(surface)
      pygame.image.save(surface, "inverted_image.png")

|

.. py:function:: invert_buffer_gpu(bgr_array)

   Perform in-place inversion of a BGR pixel buffer using the GPU.

   This function takes a contiguous 1D BGR buffer (as a NumPy array or memoryview),
   transfers it to the GPU, inverts the pixel values (255 - pixel value), and
   writes the result back into the original buffer.

   **Parameters:**

   - **bgr_array** (*numpy.ndarray* or *memoryview*): A contiguous 1D array representing the BGR pixel buffer. The buffer must be mutable and non-empty.

   **Raises:**

   - **TypeError**: If `bgr_array` is not a NumPy array or a memoryview.
   - **ValueError**: If `bgr_array` is empty (has zero length).

   **Notes:**

   - This function modifies `bgr_array` in place.
   - Uses CUDA-enabled CuPy for GPU acceleration.
   - Suitable for high-performance image processing where in-place modification is necessary.

   **Example:**

   .. code-block:: python

      import numpy as np
      import cupy as cp

      # Example BGR array
      bgr_array = np.array([255, 0, 0, 0, 255, 0, 0, 0, 255], dtype=np.uint8)

      # Convert to CuPy array
      bgr_array_gpu = cp.asarray(bgr_array)

      # Invert colors in place
      invert_buffer_gpu(bgr_array_gpu)

      # Convert back to NumPy array
      inverted_bgr_array = cp.asnumpy(bgr_array_gpu)
      print(inverted_bgr_array)  # Output: [  0 255 255 255   0 255 255 255   0]

|

.. py:function:: invert_rgb_array_inplace(rgb_array)

   Inverts the RGB values of an image (3D numpy array or memoryview) in place using GPU.

   This function performs an in-place inversion of the RGB channels in the input array.
   It operates on the assumption that the input `rgb_array` is a 3D numpy array or memoryview
   with shape (height, width, 3), where the last dimension represents the RGB color channels.

   **Note**: This operation is performed using GPU to speed up the computation.

   **Parameters:**

   - **rgb_array** (*numpy.ndarray* or *memoryview*): A 3D array or memoryview of shape (height, width, 3) containing the RGB values of the image. The dtype must be *numpy.uint8*. If `rgb_array` is a memoryview, it should be of a compatible type.

   **Raises:**

   - **TypeError**: If `rgb_array` is not a *numpy.ndarray* or *memoryview*.
   - **ValueError**: If the `rgb_array` is not of dtype *numpy.uint8* or does not have 3 dimensions. If the width or height of the array is zero.

   **Description:**

   This function directly modifies the input `rgb_array` by inverting the RGB channels using GPU-accelerated operations. The inversion is done in place, meaning the original array will be updated. The GPU is used to perform the inversion efficiently.

   **Example:**

   .. code-block:: python

      import numpy as np

      # Create a white image of shape (100, 100, 3)
      rgb_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

      # Invert the image colors in place
      invert_rgb_array_inplace(rgb_image)

      # Print the color of the first pixel, should be black [0, 0, 0]
      print(rgb_image[0, 0])

|

.. py:function:: sepia_gpu(surface_)

   Apply a Sepia shader to a Pygame surface and return a new surface with the Sepia effect applied.

   This function works with Pygame surfaces of both 32-bit and 24-bit color depth.
   It uses GPU-based computations to transform the pixel colors by applying a Sepia filter.

   The function first checks the type of the provided surface to ensure it's a valid
   Pygame Surface. Then it extracts the pixel data from the surface, applies the Sepia
   filter using a GPU-based kernel, and returns a new Pygame surface with the transformed image.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A Pygame Surface object containing the image to which the Sepia filter will be applied. The surface is expected to be in 32-bit or 24-bit format.

   **Returns:**

   - **pygame.Surface**: A new Pygame Surface object with the same dimensions and the Sepia filter applied, in the "RGB" format.

   **Notes:**

   - The function utilizes GPU acceleration to apply the Sepia effect efficiently.
   - The input surface is not modified; a new surface with the Sepia effect is returned.
   - Ensure that the Pygame surface is in a compatible format (24-bit or 32-bit) before applying the filter.

   **Example:**

   .. code-block:: python

      import pygame

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the Sepia filter
      sepia_surface = sepia_gpu(surface)

      # Save the new image
      pygame.image.save(sepia_surface, "sepia_image.png")

|

.. py:function:: sepia_rgb_array_gpu(rgb_array)

   Apply a Sepia filter in-place to an RGB image array using GPU acceleration.

   This function processes a 32-bit or 24-bit image in-place by modifying the provided
   `numpy.ndarray` of shape (w, h, 3) containing uint8 RGB values. The Sepia effect
   is applied directly to the input array without creating a new array.

   **Parameters:**

   - **rgb_array** (*numpy.ndarray*): A 3D numpy array with shape (w, h, 3), representing an image with RGB pixel values in the uint8 data type. The shape should represent the width (w), height (h), and three channels (Red, Green, Blue).

   **Returns:**

   - **None**: This function modifies the input `rgb_array` in-place and returns nothing.

   **Notes:**

   - The function utilizes GPU acceleration to apply the Sepia effect efficiently.
   - The input array is modified directly; no new array is created.
   - Ensure that the input array is of dtype `numpy.uint8` and has three dimensions representing the RGB channels.

   **Example:**

   .. code-block:: python

      import numpy as np
      import cupy as cp

      # Create a sample RGB image (100x100 pixels)
      rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

      # Convert to a CuPy array for GPU processing
      rgb_array_gpu = cp.asarray(rgb_array)

      # Apply the Sepia filter in-place
      sepia_rgb_array_gpu(rgb_array_gpu)

      # Convert back to a NumPy array if needed
      sepia_rgb_array = cp.asnumpy(rgb_array_gpu)

|

.. py:function:: sepia_buffer_gpu(grid_, block_, w, h, bgr_array, format_32=False)

   Apply a Sepia filter to a BGR or BGRA image buffer on the GPU (in-place).

   This function uses a GPU kernel to apply a Sepia effect to an image stored in either a
   BGR or BGRA format. The operation is performed on the input image buffer (`bgr_array`),
   which is a 1D array or memoryview representing the pixel data in either BGR or BGRA format.

   The function assumes the buffer is laid out as a 1D array, where each pixel consists of
   3 or 4 bytes (depending on whether it is in BGR or BGRA format). The Sepia effect is applied
   in parallel on the GPU using a CUDA kernel, and the result is stored directly in the input
   buffer.

   **Parameters:**

   - **grid_** (*tuple*): The block grid dimensions to be used for the kernel launch, typically obtained
     from a helper function such as `block_grid()`.
   - **block_** (*tuple*): The block dimensions to be used for the kernel launch, typically obtained
     from a helper function such as `block_grid()`.
   - **w** (*Py_ssize_t*): The width of the image, in pixels. Must be greater than 0.
   - **h** (*Py_ssize_t*): The height of the image, in pixels. Must be greater than 0.
   - **bgr_array** (*unsigned char [::1]*): A 1D array or memoryview containing the image pixel data. This array represents
     the image in either BGR (3 channels) or BGRA (4 channels) format, with each pixel
     occupying 3 or 4 consecutive bytes.
   - **format_32** (*bint*, optional): A boolean flag indicating whether the input array is in BGRA format (True) or BGR
     format (False). Default is False (BGR format).

   **Returns:**

   - **None**: This function modifies the input `bgr_array` in-place and returns nothing.

   **Raises:**

   - **TypeError**: If `bgr_array` is not a valid numpy.ndarray or memoryview.
   - **ValueError**: If the dimensions of the input image or the buffer length do not match
     the expected values based on the width, height, and pixel format.

   **Notes:**

   - The function utilizes GPU acceleration to apply the Sepia effect efficiently.
   - The input array is modified directly; no new array is created.
   - Ensure that the input array is of dtype `numpy.uint8` and has three or four dimensions representing the RGB or RGBA channels.

   **Example:**

   .. code-block:: python

      import numpy as np
      import cupy as cp

      # Create a sample BGR image (100x100 pixels)
      bgr_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

      # Convert to a CuPy array for GPU processing
      bgr_array_gpu = cp.asarray(bgr_array)

      # Define grid and block dimensions
      grid = (32, 32)
      block = (16, 16)

      # Apply the Sepia filter in-place
      sepia_buffer_gpu(grid, block, 100, 100, bgr_array_gpu)

      # Convert back to a NumPy array if needed
      sepia_bgr_array = cp.asnumpy(bgr_array_gpu)

|

.. py:function:: bpf_gpu(surface_, threshold_=128)

   Apply a Bright Pass Filter (BPF) effect to a Pygame surface using GPU acceleration.

   This function applies a Bright Pass Filter to an image represented by a Pygame.Surface.
   The filter retains pixels with brightness values above a specified threshold and sets
   others to black. The operation is performed on the GPU to enhance performance.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input surface (image) on which the Bright Pass Filter effect will be applied.
   - **threshold_** (*int*, optional): The brightness threshold in the range [0, 255]. Pixels with a brightness value
     greater than or equal to the threshold will retain their color, while those below will be set to black.
     Default is 128.

   **Returns:**

   - **pygame.Surface**: A new Pygame.Surface with the Bright Pass Filter effect applied, in RGB format, with the same width and height as the input surface.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface` type.
   - **ValueError**: If there is an issue referencing the surface pixels into a 3D array, or if the surface dimensions are invalid.

   **Notes:**

   - The function utilizes GPU acceleration to apply the Bright Pass Filter efficiently.
   - The input surface is not modified; a new surface with the Bright Pass Filter effect is returned.
   - Ensure that the Pygame surface is in a compatible format (24-bit or 32-bit) before applying the filter.

   **Example:**

   .. code-block:: python

      import pygame

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the Bright Pass Filter with a threshold of 150
      threshold_value = 150
      bpf_surface = bpf_gpu(surface, threshold_value)

      # Save the new image
      pygame.image.save(bpf_surface, "bpf_image.png")

|

.. py:function:: bpf_inplace_gpu(grid_, block_, surface_, threshold_=128)

   Apply a Bright Pass Filter (BPF) to a Pygame surface in-place using GPU acceleration.

   This function applies a Bright Pass Filter effect to an image on the GPU. The filter retains
   pixels with brightness values greater than or equal to a specified threshold, setting others to black.
   The operation is done in-place, modifying the input surface directly.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input surface (image) on which the Bright Pass Filter effect will be applied. The surface should be in RGB format.
   - **grid_** (*tuple*): Specifies the grid configuration for the GPU kernel. This determines how the work is divided across multiple threads in the GPU. Grid dimensions should match the texture and array sizes.
   - **block_** (*tuple*): Specifies the block configuration for the GPU kernel. This defines the number of threads within a block. The maximum number of threads per block is 1024. Block size should be chosen to optimize performance on the hardware.
   - **threshold_** (*int*, optional): The brightness threshold (in the range [0, 255]) that determines which pixels are kept. Pixels with a brightness value greater than or equal to the threshold remain unchanged, while pixels below the threshold are set to black. Default is 128.

   **Returns:**

   - **None**: The function modifies the input `surface_` in-place and returns nothing.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface` type.
   - **ValueError**: If there is an issue referencing the surface pixels into a 3D array, or if the surface dimensions are invalid.

   **Notes:**

   - The function utilizes GPU acceleration to apply the Bright Pass Filter efficiently.
   - The input surface is modified directly; no new surface is created.
   - Ensure that the Pygame surface is in a compatible format (24-bit or 32-bit) before applying the filter.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Define grid and block dimensions
      grid = (32, 32)
      block = (16, 16)

      # Apply the Bright Pass Filter with a threshold of 150
      threshold_value = 150
      bpf_inplace_gpu(grid, block, surface, threshold_value)

      # Save the new image
      pygame.image.save(surface, "bpf_image.png")

|

.. py:function:: bpf_buffer_gpu(grid, block, w, h, bgr_array, threshold_=128, format_32=False)

   Apply a band-pass filter (BPF) to an image buffer using GPU acceleration.

   This function processes an image buffer represented by a 1D array or memoryview in either BGR or BGRA format.
   The BPF retains pixels with brightness values within a specified range and sets others to black.
   The operation is performed on the GPU to enhance performance.

   **Parameters:**

   - **grid** (*tuple*): Grid size for GPU kernel execution (blocks per grid).
   - **block** (*tuple*): Block size for GPU kernel execution (threads per block).
   - **w** (*int*): Width of the image in pixels.
   - **h** (*int*): Height of the image in pixels.
   - **bgr_array** (*numpy.ndarray* or *memoryview*): Input image buffer in BGR or BGRA format.
   - **threshold_** (*int*, optional): Threshold value used for filtering (default: 128).
   - **format_32** (*bool*, optional): If True, treats the input buffer as BGRA (4 bytes per pixel). If False, treats it as BGR (3 bytes per pixel).

   **Returns:**

   - **None**: The function modifies `bgr_array` in-place with the filtered image.

   **Raises:**

   - **TypeError**: If `bgr_array` is not a numpy.ndarray or memoryview.
   - **ValueError**: If `bgr_array` is empty or does not match the expected size.

   **Notes:**

   - The function utilizes GPU acceleration to apply the BPF efficiently.
   - The input array is modified directly; no new array is created.
   - Ensure that the input array is of dtype `numpy.uint8` and has three or four dimensions representing the RGB or RGBA channels.

   **Example:**

   .. code-block:: python

      import numpy as np
      import cupy as cp

      # Create a sample BGR image (100x100 pixels)
      bgr_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

      # Convert to a CuPy array for GPU processing
      bgr_array_gpu = cp.asarray(bgr_array)

      # Define grid and block dimensions
      grid = (32, 32)
      block = (16, 16)

      # Apply the BPF with a threshold of 150
      threshold_value = 150
      bpf_buffer_gpu(grid, block, 100, 100, bgr_array_gpu, threshold_value)

      # Convert back to a NumPy array if needed
      filtered_bgr_array = cp.asnumpy(bgr_array_gpu)

|

.. py:function:: grayscale_gpu(surface_)

   Converts a Pygame surface to grayscale using GPU acceleration.

   This function takes a Pygame surface in 24-bit or 32-bit format, processes
   it on the GPU to convert it to grayscale, and returns a new 24-bit grayscale surface.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A Pygame surface with a 24-bit or 32-bit color format.

   **Returns:**

   - **pygame.Surface**: A new 24-bit grayscale Pygame surface.

   **Raises:**

   - **TypeError**: If `surface_` is not a Pygame surface.
   - **ValueError**: If the surface cannot be converted into a 3D array view, or if the surface width or height is zero.

   **Notes:**

   - The function utilizes GPU acceleration to perform the grayscale conversion efficiently.
   - The input surface is not modified; a new surface with the grayscale effect is returned.
   - Ensure that the Pygame surface is in a compatible format (24-bit or 32-bit) before applying the filter.

   **Example:**

   .. code-block:: python

      import pygame

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Convert the image to grayscale
      grayscale_surface = grayscale_gpu(surface)

      # Save the new image
      pygame.image.save(grayscale_surface, "grayscale_image.png")

|

.. py:function:: grayscale_lum_gpu(surface_)

   Converts a Pygame surface to grayscale using GPU acceleration with a luminance-based method.

   This function takes a Pygame surface in 24-bit or 32-bit format, processes
   it on the GPU using a luminance-based grayscale conversion, and returns
   a new 24-bit grayscale Pygame surface.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A Pygame surface with a 24-bit or 32-bit color format.

   **Returns:**

   - **pygame.Surface**: A new 24-bit grayscale Pygame surface.

   **Raises:**

   - **TypeError**: If `surface_` is not a Pygame surface.
   - **ValueError**: If the surface cannot be converted into a 3D array view, or if the surface width or height is zero.

   **Notes:**

   - The function utilizes GPU acceleration to perform the grayscale conversion efficiently.
   - The input surface is not modified; a new surface with the grayscale effect is returned.
   - Ensure that the Pygame surface is in a compatible format (24-bit or 32-bit) before applying the filter.

   **Example:**

   .. code-block:: python

      import pygame

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Convert the image to grayscale
      grayscale_surface = grayscale_lum_gpu(surface)

      # Save the new image
      pygame.image.save(grayscale_surface, "grayscale_image.png")

|

.. py:function:: median_generic_filter_gpu(surface_, size_=5)

   Apply a median filter to a Pygame surface using GPU acceleration.

   This function utilizes a generic median filter, which processes each
   pixel based on neighboring values. The filter operates on 24-bit and
   32-bit surfaces by applying a kernel function.

   **Performance Consideration:**
   The median filter uses a kernel with a buffer of type `double`, which
   may result in lower performance compared to specialized GPU-based median
   filtering methods.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input Pygame surface, expected to be in 24-bit or 32-bit format.
   - **size_** (*int*, optional): The size of the neighborhood window used for the median filter. Must be greater than zero. Default is 5.

   **Returns:**

   - **pygame.Surface**: A new Pygame surface with the median filter applied.

   **Raises:**

   - **ValueError**: If `size_` is less than or equal to zero, or if the input surface has zero width or height.
   - **TypeError**: If `surface_` is not a valid `pygame.Surface`.

   **Notes:**

   - This function leverages CuPy (`cp.asarray`) for GPU-accelerated computation.
   - The filter operates on each color channel (R, G, B) separately.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the median filter with a window size of 5
      filtered_surface = median_generic_filter_gpu(surface, size_=5)

      # Save the new image
      pygame.image.save(filtered_surface, "filtered_image.png")

|

.. py:function:: median_filter_gpu(surface_, size_=5)

   Apply a median filter to a Pygame surface using GPU acceleration.

   The median filter is a non-linear filter that reduces noise by replacing
   each pixel with the median value of its surrounding neighbors.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input Pygame surface to be processed.
   - **size_** (*int*, optional): The size of the neighborhood considered for median filtering. Must be greater than zero. Default is 5.

   **Returns:**

   - **pygame.Surface**: A new Pygame surface with the median filter applied.

   **Raises:**

   - **AssertionError**: If `size_` is not greater than zero.
   - **ValueError**: If `surface_` is not a valid Pygame surface, or if the surface cannot be referenced as a 3D array, or if the surface width or height is zero.

   **Notes:**

   - This function leverages CuPy (`cp.asarray`) for GPU-accelerated computation.
   - The filter operates on each color channel (R, G, B) separately.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the median filter with a window size of 5
      filtered_surface = median_filter_gpu(surface, size_=5)

      # Save the new image
      pygame.image.save(filtered_surface, "filtered_image.png")

|

.. py:function:: gaussian_5x5_gpu(surface_)

   Apply a 5x5 Gaussian blur to an image using GPU-accelerated processing.

   This function convolves each RGB channel of the given surface with a
   5x5 Gaussian blur kernel to create a smoothing effect while preserving edges.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A 24-bit or 32-bit Pygame surface to be blurred.

   **Returns:**

   - **pygame.Surface**: A new Pygame surface with the applied Gaussian blur in 24-bit format.

   **Raises:**

   - **TypeError**: If `surface_` is not a valid Pygame surface.
   - **ValueError**: If the surface dimensions are invalid or pixel data cannot be accessed.

   **Notes:**

   - The function uses `pixels3d(surface_)` to extract the pixel array.
   - The computation is performed on the GPU for efficiency.
   - `gaussian_5x5_cupy` is called to apply the blur effect.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the 5x5 Gaussian blur
      blurred_surface = gaussian_5x5_gpu(surface)

      # Save the new image
      pygame.image.save(blurred_surface, "blurred_image.png")

|

.. py:function:: gaussian_3x3_gpu(surface_)

   Apply a 3x3 Gaussian blur to a Pygame surface using GPU acceleration.

   This function converts a given `pygame.Surface` to a GPU-compatible array,
   applies a 3x3 Gaussian blur filter to each color channel (R, G, B), and
   returns a new blurred `pygame.Surface`.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A 24-bit or 32-bit `pygame.Surface` representing the input image.

   **Returns:**

   - **pygame.Surface**: A new `pygame.Surface` with the Gaussian blur effect applied.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If the surface has invalid dimensions or cannot be converted into an array.

   **Notes:**

   - The function assumes that `gaussian_kernel_3x3` is predefined.
   - `cp.cuda.Stream.null.synchronize()` ensures the GPU operations complete before returning.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the 3x3 Gaussian blur
      blurred_surface = gaussian_3x3_gpu(surface)

      # Save the new image
      pygame.image.save(blurred_surface, "blurred_image.png")

|

.. py:function:: sobel_gpu(surface_)

   Apply the Sobel edge detection filter to a grayscale image using GPU acceleration.

   This function assumes the input image is grayscale, meaning the same intensity
   is present in all three RGB channels. It applies the Sobel filter using only the red
   channel (`R`), though the green (`G`) or blue (`B`) channels could also be used.

   If the input image is not truly grayscale, the Sobel effect may vary slightly because
   the RGB channels might have different intensities.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A 24-bit or 32-bit `pygame.Surface` representing the input image.

   **Returns:**

   - **pygame.Surface**: A new `pygame.Surface` with the Sobel edge detection effect applied in 24-bit format.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If the surface has invalid dimensions or cannot be converted into an array.

   **Notes:**

   - The function assumes that `sobel_cupy()` is implemented to process the GPU array.
   - The final image retains the `RGB` format, but only the red channel is modified.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the Sobel edge detection filter
      edge_surface = sobel_gpu(surface)

      # Save the new image
      pygame.image.save(edge_surface, "edge_image.png")

|

.. py:function:: prewitt_gpu(surface_)

   Apply the Prewitt edge detection algorithm to a grayscale image using GPU acceleration.

   This function assumes the input image is grayscale, meaning the same intensity
   is present in all three RGB channels. It applies the Prewitt filter using only the red
   channel (`R`), though the green (`G`) or blue (`B`) channels could also be used.

   If the input image is not truly grayscale, the Prewitt effect may vary slightly because
   the RGB channels might have different intensities.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A 24-bit or 32-bit `pygame.Surface` representing the input image.

   **Returns:**

   - **pygame.Surface**: A new `pygame.Surface` with the Prewitt edge detection effect applied in 24-bit format.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If the surface has invalid dimensions or cannot be converted into an array.

   **Notes:**

   - The function assumes that `prewitt_cupy()` is implemented to process the GPU array.
   - The final image retains the `RGB` format, but only the red channel is modified.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the Prewitt edge detection filter
      edge_surface = prewitt_gpu(surface)

      # Save the new image
      pygame.image.save(edge_surface, "edge_image.png")

|

.. py:function:: canny_gpu(surface_)

   Apply the Canny edge detection algorithm to a grayscale image using GPU acceleration.

   This function assumes the input image is grayscale, meaning the same intensity
   is present in all three RGB channels. It applies the Canny edge detection algorithm
   using only the red channel (`R`), though the green (`G`) or blue (`B`) channels could
   also be used. If the input image is not truly grayscale, the Canny effect may vary
   slightly because the RGB channels might have different intensities.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A 24-bit or 32-bit `pygame.Surface` representing the input image.

   **Returns:**

   - **pygame.Surface**: A new `pygame.Surface` with the Canny edge detection effect applied in 24-bit format.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If the surface has invalid dimensions or cannot be converted into an array.

   **Notes:**

   - The function assumes that `canny_cupy()` is implemented to process the GPU array.
   - The final image retains the `RGB` format, but only the red channel is modified.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the Canny edge detection filter
      edge_surface = canny_gpu(surface)

      # Save the new image
      pygame.image.save(edge_surface, "edge_image.png")

|

.. py:function:: color_reduction_gpu(surface_, color_number=8)

   Apply a color reduction effect to an image using GPU acceleration.

   This function reduces the number of unique colors in the image by quantizing each RGB channel.
   For example, if `color_number=8`, each channel will have 8 distinct levels (0, 32, 64, ..., 255),
   resulting in a total of 8^3 possible colors.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input surface (image) to which the color reduction will be applied.
   - **color_number** (*int*, optional): The number of colors (levels) to reduce each RGB channel to. The default value is 8. The number of colors in the palette will be `color_number^3`.

   **Returns:**

   - **pygame.Surface**: A new surface with the color reduction effect applied, in 24-bit format.

   **Raises:**

   - **TypeError**: If the `surface_` argument is not a valid `pygame.Surface` instance.
   - **ValueError**: If the `color_number` is less than or equal to 0 or if the surface has invalid dimensions.

   **Notes:**

   - The function utilizes GPU acceleration for efficient processing.
   - The color reduction is achieved by quantizing each RGB color channel.
   - The function works with 32-bit and 24-bit images (Pygame Surface format).

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the color reduction with a palette size of 8
      reduced_surface = color_reduction_gpu(surface, color_number=8)

      # Save the new image
      pygame.image.save(reduced_surface, "reduced_image.png")

|

.. py:function:: hsv_gpu(surface_, val_, grid_, block_)

   Apply a hue rotation effect to an image using GPU acceleration.

   This function modifies the hue of each pixel in the image while preserving its
   saturation and brightness, effectively shifting colors while maintaining overall
   image structure.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input surface (image) on which the hue rotation effect will be applied.
   - **val_** (*float*): The hue shift value, which must be within the range [0.0, 1.0].
     - `0.0` corresponds to a -180° shift (full backward rotation).
     - `0.5` represents 0° shift (no change).
     - `1.0` corresponds to a +180° shift (full forward rotation).
   - **grid_** (*tuple*): Specifies the CUDA grid dimensions for kernel execution.
     - Example: `(grid_y, grid_x)`, e.g., `(25, 25)`.
     - The grid size should be tuned according to the texture and array sizes.
   - **block_** (*tuple*): Specifies the CUDA block dimensions for kernel execution.
     - Example: `(block_y, block_x)`, e.g., `(32, 32)`.
     - The total number of threads (`block_x * block_y`) must not exceed 1024 due to GPU hardware limitations.

   **Returns:**

   - **pygame.Surface**: A new surface containing the hue-modified image, returned in 24-bit RGB format.

   **Raises:**

   - **TypeError**: If `surface_` is not a valid `pygame.Surface` instance.
   - **ValueError**: If the input surface has zero width or height, if `val_` is out of the valid range [0.0, 1.0], or if the surface pixels cannot be referenced as a 3D array.

   **Notes:**

   - The input image must be in RGB format before applying this transformation.
   - The function performs hue rotation using GPU acceleration with CuPy for optimized performance.
   - The transformation works by converting RGB to HSV, modifying the H (hue) channel, and then converting it back to RGB.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Define grid and block dimensions
      grid = (25, 25)
      block = (32, 32)

      # Apply the hue rotation with a shift value of 0.5 (no change)
      rotated_surface = hsv_gpu(surface, val_=0.5, grid_=grid, block_=block)

      # Save the new image
      pygame.image.save(rotated_surface, "rotated_image.png")

|

.. py:function:: mult_downscale_gpu(gpu_array)

   Perform bloom downscaling by progressively reducing image resolution.

   This function downscales an input GPU-based image into four progressively
   smaller sub-arrays with resolutions reduced by factors of 2, 4, 8, and 16.
   It is compatible with 24-bit and 32-bit images (RGB format, uint8).

   **Parameters:**

   - **gpu_array** (*cupy.ndarray*): A 3D CuPy array with shape (height, width, 3) and dtype uint8,
     representing an RGB image stored on the GPU.

   **Returns:**

   - **tuple of cupy.ndarray**: A tuple containing four downscaled versions of the input image
     with shapes (H/2, W/2, 3), (H/4, W/4, 3), (H/8, W/8, 3), and (H/16, W/16, 3).

   **Notes:**

   - The function performs downscaling using `cupyx.scipy.ndimage.zoom`,
     applying nearest-neighbor interpolation (`order=0`) to preserve hard edges.
   - Pixels outside the boundaries are filled with zero (`mode='constant', cval=0.0`).
   - GPU synchronization is performed to ensure all CUDA operations complete before returning.

   **Example:**

   .. code-block:: python

      import cupy as cp

      # Assume gpu_array is a 3D CuPy array representing an RGB image
      downscaled_images = mult_downscale_gpu(gpu_array)

      # Access the downscaled images
      image_2x = downscaled_images[0]
      image_4x = downscaled_images[1]
      image_8x = downscaled_images[2]
      image_16x = downscaled_images[3]

|

.. py:function:: upscale_c(gpu_array_, new_width, new_height, order_=0)

   Upscale an image on the GPU using interpolation.

   This function resizes a GPU-based image to the specified `new_width` and `new_height`
   using the specified interpolation order. The resizing is performed using
   `cupyx.scipy.ndimage.zoom`, which supports different interpolation methods.

   **Parameters:**

   - **gpu_array_** (*cupy.ndarray*): A 3D GPU array (height x width x 3) representing an RGB image in `uint8` format.
   - **new_width** (*int*): The desired width of the upscaled image.
   - **new_height** (*int*): The desired height of the upscaled image.
   - **order_** (*int*, optional, default=0): The interpolation order for resizing:
     - `0`: Nearest-neighbor interpolation.
     - `1`: Bilinear interpolation.
     - `3`: Bicubic interpolation.
     Higher orders provide smoother results but increase computation time.

   **Returns:**

   - **cupy.ndarray**: The upscaled image as a GPU-based array with shape (`new_height`, `new_width`, 3).

   **Notes:**

   - The function ensures that the input array is of type `uint8` before processing.
   - The `mode='constant', cval=0.0` ensures that pixels outside the boundaries
     are treated as zero during interpolation.
   - GPU synchronization (`cp.cuda.Stream.null.synchronize()`) is performed to
     ensure all operations complete before returning the result.

   **Example:**

   .. code-block:: python

      import cupy as cp

      # Assume gpu_array is a 3D CuPy array representing an RGB image
      upscaled_image = upscale_c(gpu_array, new_width=800, new_height=600, order_=1)

|

.. py:function:: bloom_gpu(surface_, threshold_=128, fast_=True, flag_=pygame.BLEND_RGB_ADD, factor_=2)

   Apply a Bloom Effect to a Pygame Surface using GPU Acceleration.

   This function enhances bright regions of an image to create a glowing effect (bloom)
   using a multi-step process that involves:
   - Downscaling the image into progressively smaller sub-surfaces.
   - Applying a bright-pass filter to isolate bright areas.
   - Blurring the bright regions using a Gaussian filter.
   - Upscaling and blending the processed layers back into the original surface.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input surface to apply the bloom effect on.
   - **threshold_** (*int*, optional, default=128): The brightness threshold for the bright-pass filter.
     Pixels with intensity above this value contribute to the bloom effect. Must be in range [0, 255].
   - **fast_** (*bool*, optional, default=True): When `True`, only the lowest-resolution downsample (x16) is used for bloom.
     When `False`, multiple downsampled layers (x2, x4, x8, x16) are processed and blended for a more detailed effect.
   - **flag_** (*int*, optional, default=pygame.BLEND_RGB_ADD): The blending mode used when merging the bloom effect back onto the original surface.
     Common flags include:
     - `pygame.BLEND_RGB_ADD` (default)
     - `pygame.BLEND_RGB_MULT`, etc.
   - **factor_** (*int*, optional, default=2): Determines how much the texture is downscaled.
     Must be in the range [0, 4], corresponding to downscaling by:
     - 1 (no reduction)
     - 2 (half-size)
     - 4 (quarter-size)
     - 8 (eighth-size)

   **Returns:**

   - **pygame.Surface**: The input surface blended with the bloom effect.

   **Raises:**

   - **ValueError**: If the input image is too small to be processed.

   **Notes:**

   - The function utilizes GPU acceleration for efficient processing.
   - The bloom effect is achieved through a series of downscaling, filtering, and blending operations.
   - The `pygame.BLEND_RGB_ADD` flag is used to add the bloom effect to the original surface.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the bloom effect
      bloomed_surface = bloom_gpu(surface, threshold_=150, fast_=False, factor_=3)

      # Save the new image
      pygame.image.save(bloomed_surface, "bloomed_image.png")

|

.. py:function:: bloom_array(gpu_array_, threshold_=128, fast_=True, flag_=pygame.BLEND_RGB_ADD, mask_=None)

   Apply a Bloom Effect to an Image Represented as a GPU Array.

   This function performs a bloom effect on a given `gpu_array_`, which represents an image on the GPU.
   The process involves:
   - Downscaling the image into progressively smaller versions (x2, x4, x8, x16).
   - Applying a bright-pass filter to isolate high-intensity pixels.
   - Blurring the bright regions to create a glow effect.
   - Upscaling and blending the processed images to achieve the final bloom effect.

   **Parameters:**

   - **gpu_array_** (*cupy.ndarray*): A 3D array representing the image in GPU memory. Must have dtype `cupy.uint8`.
   - **threshold_** (*int*, optional, default=128): The brightness threshold for the bright-pass filter. Pixels above this value contribute to bloom. Must be within [0, 255].
   - **fast_** (*bool*, optional, default=True): When `True`, only the lowest-resolution downsample (x16) is used, reducing processing time. When `False`, multiple levels (x2, x4, x8, x16) are blended for a richer bloom effect.
   - **flag_** (*int*, optional, default=pygame.BLEND_RGB_ADD): The blending mode used when merging the bloom effect layers.
   - **mask_** (*optional*, default=None): Unused parameter, reserved for future implementations.

   **Returns:**

   - **tuple of pygame.Surface**: The processed bloom effect surfaces at different downscaling levels (s2, s4, s8, s16). If `fast_` is enabled, only `s16` is used.

   **Notes:**

   - The function utilizes GPU acceleration for efficient processing.
   - The bloom effect is achieved through a series of downscaling, filtering, and blending operations.
   - The `pygame.BLEND_RGB_ADD` flag is used to add the bloom effect to the original surface.

   **Example:**

   .. code-block:: python

      import cupy as cp
      import pygame

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Convert Pygame surface to CuPy array
      gpu_array = cp.asarray(pygame.surfarray.pixels3d(surface))

      # Apply the bloom effect
      bloomed_surfaces = bloom_array(gpu_array, threshold_=150, fast_=False, flag_=pygame.BLEND_RGB_ADD)

      # Convert back to Pygame surfaces
      bloomed_images = [pygame.surfarray.make_surface(cp.asnumpy(bloom)) for bloom in bloomed_surfaces]

      # Save the new images
      for i, bloomed_image in enumerate(bloomed_images):
          pygame.image.save(bloomed_image, f"bloomed_image_{i}.png")

|

.. py:function:: cartoon_gpu(surface_, sobel_threshold_=128, median_kernel_=2, color_=8, contour_=False, flag_=BLEND_RGB_ADD)

   Apply a cartoon effect to a given surface using GPU acceleration.

   This function processes an image to generate a cartoon-like effect by applying
   edge detection, median filtering, and color quantization. The effect can also
   include contour outlining if enabled.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input image surface to be processed.
   - **sobel_threshold_** (*int*, optional, default=128): The threshold value for the Sobel edge detection filter.
   - **median_kernel_** (*int*, optional, default=2): Kernel size for median filtering (used to reduce noise while preserving edges).
   - **color_** (*int*, optional, default=8): Number of maximum colors to be used in the cartoon effect (color reduction).
   - **contour_** (*bool*, optional, default=False): Whether to draw contours on the edges detected in the image.
   - **flag_** (*int*, optional, default=BLEND_RGB_ADD): Blending mode used to combine the effect with the original image.

   **Returns:**

   - **pygame.Surface**: The transformed surface with the cartoon effect applied.

   **Notes:**

   - The function utilizes GPU acceleration for efficient processing.
   - The cartoon effect is achieved through a series of edge detection, filtering, and blending operations.
   - The `BLEND_RGB_ADD` flag is used to add the cartoon effect to the original surface.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the cartoon effect
      cartoon_surface = cartoon_gpu(surface, sobel_threshold_=150, median_kernel_=3, color_=8, contour_=True, flag_=pygame.BLEND_RGB_ADD)

      # Save the new image
      pygame.image.save(cartoon_surface, "cartoon_image.png")

|

.. py:function:: blending_gpu(source_, destination_, percentage_)

   Blend a source texture toward a destination texture (transition effect).

   This function blends a source Pygame surface onto a destination surface, creating a transition effect
   based on the specified percentage. The blending is performed using the following calculations:
   - **Alpha**: `outA = SrcA + DstA * (1 - SrcA)`
   - **RGB**: `outRGB = SrcRGB + DstRGB * (1 - SrcA)`

   **Parameters:**

   - **source_** (*pygame.Surface*): The source surface to be blended.
   - **destination_** (*pygame.Surface*): The destination surface onto which the source will be blended.
   - **percentage_** (*float*): The blending percentage, a value between 0.0 and 100.0, representing the proportion of the source to blend with the destination.

   **Returns:**

   - **pygame.Surface**: A new 24-bit Pygame surface with the blended result.

   **Notes:**

   - The function assumes that the source and destination surfaces are of the same size.
   - The output surface is converted for fast blitting using the `convert()` method.
   - The blending mode used is `BLEND_RGB_ADD`, which adds the source color to the destination color. This mode can be adjusted by changing the `flag_` parameter.

   **Example:**

   .. code-block:: python

      import pygame

      # Initialize Pygame
      pygame.init()

      # Load source and destination images
      source = pygame.image.load('source_image.png')
      destination = pygame.image.load('destination_image.png')

      # Apply blending with 50% source
      blended_surface = blending_gpu(source, destination, 50.0)

      # Save the blended image
      pygame.image.save(blended_surface, 'blended_image.png')

|

.. py:function:: sharpen_gpu(surface_)

   Apply a sharpening filter to the given pygame.Surface image using a generic filter kernel.

   The function sharpens the input image by applying a sharpening kernel to each RGB channel
   (red, green, blue) independently. The sharpened image is returned as a 24-bit pygame.Surface
   format.

   **Parameters:**

   - **surface_** (*pygame.Surface*): A pygame surface object that represents the input image to be sharpened.

   **Returns:**

   - **pygame.Surface**: A new pygame surface with the sharpened image in 24-bit format (RGB).

   **Raises:**

   - **ValueError**: If the input surface is not a valid pygame.Surface or has zero dimensions.

   **Notes:**

   - The function utilizes GPU acceleration for efficient processing.
   - The sharpening effect is achieved through convolution with a sharpening kernel.
   - The output surface is converted for fast blitting using the `convert()` method.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Apply the sharpening filter
      sharpened_surface = sharpen_gpu(surface)

      # Save the sharpened image
      pygame.image.save(sharpened_surface, "sharpened_image.png")

|

.. py:function:: ripple_effect_gpu(grid, block, w, h, previous, current, texture_array, background_array)

   Apply a water drop (ripple) effect to a texture using GPU acceleration.

   This function uses a CUDA kernel (`ripple_kernel`) to compute the ripple effect on the texture.
   It works by manipulating the `previous` and `current` state of the ripple effect and applying
   it to the texture. The effect is computed in parallel on the GPU, making it efficient for
   handling large textures or simulations.

   **Parameters:**

   - **grid** (*tuple*): Tuple defining the grid size for CUDA kernel execution, e.g., (25, 25).
     The grid size should match the texture size for efficient parallel processing.
   - **block** (*tuple*): Tuple defining the block size for CUDA kernel execution, e.g., (32, 32).
     The maximum number of threads is 1024 (block_x * block_y).
   - **w** (*int*): The width of the texture and arrays.
   - **h** (*int*): The height of the texture and arrays.
   - **previous** (*cupy.ndarray*): A CuPy array of shape (w, h), containing the previous state of the ripple effect (float32).
   - **current** (*cupy.ndarray*): A CuPy array of shape (w, h), containing the current state of the ripple effect (float32).
   - **texture_array** (*cupy.ndarray*): A CuPy array of shape (w, h, 3), containing the source texture in RGB (uint8).
   - **background_array** (*cupy.ndarray*): A CuPy array of shape (w, h, 3), representing the background texture to apply
     the effect to (uint8).

   **Returns:**

   - **tuple**: A tuple containing two CuPy arrays, the updated `previous` and `current` states of the ripple effect.

   **Example:**

   .. code-block:: python

      import cupy as cp

      # Define grid and block dimensions
      grid = (25, 25)
      block = (32, 32)
      w, h = 512, 512

      # Initialize previous and current ripple states
      previous = cp.zeros((w, h), dtype=cp.float32)
      current = cp.zeros((w, h), dtype=cp.float32)

      # Initialize texture and background arrays
      texture_array = cp.random.randint(0, 256, (w, h, 3), dtype=cp.uint8)
      background_array = cp.zeros((w, h, 3), dtype=cp.uint8)

      # Apply ripple effect
      updated_previous, updated_current = ripple_effect_gpu(grid, block, w, h, previous, current, texture_array, background_array)

   **Notes:**

   - The grid and block sizes must be chosen appropriately based on the texture size and available GPU resources.
   - The kernel computation is done on the GPU using CuPy for faster execution.

|

.. py:function:: sharpen1_gpu(surface_, grid_, block_)

   Sharpen an image using a raw kernel.

   This function sharpens the input image by applying a sharpening kernel to each RGB channel
   (red, green, blue) independently. The borders are not computed with the kernel (value = 0).

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input image surface to be sharpened.
   - **grid_** (*tuple*): Grid dimensions for CUDA kernel execution, e.g., (25, 25). The grid values and block values must match the texture and array sizes.
   - **block_** (*tuple*): Block dimensions for CUDA kernel execution, e.g., (32, 32). Maximum threads are 1024. Max threads = block_x * block_y.

   **Returns:**

   - **pygame.Surface**: A new Pygame surface with the sharpened image in 24-bit format (RGB).

   **Notes:**

   - The function utilizes GPU acceleration for efficient processing.
   - The sharpening effect is achieved through convolution with a sharpening kernel.
   - The output surface is converted for fast blitting using the `convert()` method.

   **Example:**

   .. code-block:: python

      import pygame
      import cupy as cp

      # Initialize Pygame
      pygame.init()

      # Load an image
      surface = pygame.image.load("image.png").convert()

      # Define grid and block dimensions
      grid = (25, 25)
      block = (32, 32)

      # Apply the sharpening filter
      sharpened_surface = sharpen1_gpu(surface, grid, block)

      # Save the sharpened image
      pygame.image.save(sharpened_surface, "sharpened_image.png")

|

.. py:function:: mirroring_gpu(surface_, grid_, block_, format_=False)

   Apply a mirror effect to an image using GPU acceleration.

   This function creates a mirrored version of an image represented as a `pygame.Surface`.
   It supports 32-bit and 24-bit image formats and outputs a 24-bit format image.
   The mirror orientation (horizontal or vertical) is controlled using the `format_` parameter.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input image as a `pygame.Surface` object.
   - **grid_** (*tuple*): Defines the grid dimensions as `(grid_y, grid_x)`, e.g., `(25, 25)`.
     The grid and block dimensions must match the texture and array sizes.
   - **block_** (*tuple*): Specifies the block dimensions as `(block_y, block_x)`, e.g., `(32, 32)`.
     The maximum number of threads per block is 1024, following the condition:
     `Max threads = block_x * block_y`.
   - **format_** (*bool*, optional, default=`False`): Determines the mirror orientation:
     - `False` (0) → Horizontal mirroring (default).
     - `True` (1) → Vertical mirroring.

   **Returns:**

   - **pygame.Surface**: A 24-bit `pygame.Surface` with the applied mirror effect.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface` object.
   - **ValueError**: If `surface_` has an invalid format or dimensions (`w` or `h` is 0).
   - **ValueError**: If `surface_` cannot be referenced as a 3D array.

   **Notes:**

   - The function extracts pixel data from `surface_` and converts it into a 3D array.
   - The `mirroring_cupy` function performs GPU-accelerated mirroring and returns the processed image.

|

.. py:function:: saturation_gpu(surface_, grid_, block_, val_=1.0)

   Adjusts the saturation level of a given image using GPU acceleration.

   This function modifies the saturation of an image represented as a `pygame.Surface`.
   It supports 32-bit and 24-bit image formats, producing an output in 24-bit format.
   The saturation adjustment value must be within the range [-1.0, 1.0].

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input image as a `pygame.Surface` object.
   - **grid_** (*tuple*): Defines the grid dimensions as `(grid_y, grid_x)`, e.g., `(25, 25)`.
     The grid and block dimensions must match the texture and array sizes.
   - **block_** (*tuple*): Specifies the block dimensions as `(block_y, block_x)`, e.g., `(32, 32)`.
     The maximum number of threads per block is 1024, following the condition:
     `Max threads = block_x * block_y`.
   - **val_** (*float*, optional, default=`1.0`): Saturation adjustment level in the range `[-1.0, 1.0]`.
     - `-1.0` → Completely desaturated (grayscale).
     - `0.0` → No change in saturation.
     - `1.0` → Fully saturated.

   **Returns:**

   - **pygame.Surface**: A 24-bit `pygame.Surface` with the adjusted saturation level.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface` object.
   - **ValueError**: If `surface_` has an invalid format or dimensions (`w` or `h` is 0).
   - **ValueError**: If `val_` is outside the range `[-1.0, 1.0]`.

   **Notes:**

   - The function extracts pixel data from `surface_` and converts it into a 3D array.
   - The `saturation_cupy` function performs GPU-accelerated saturation adjustment
     and returns the processed image.

|

.. py:function:: bilateral_gpu(surface_, kernel_size_)

   Apply a bilateral filter to a 32-bit or 24-bit image using the GPU.

   A bilateral filter is a non-linear, edge-preserving, and noise-reducing
   smoothing filter. It replaces the intensity of each pixel with a weighted
   average of nearby pixels, where the weights are determined by both spatial
   proximity and intensity differences, following a Gaussian distribution.

   The filter parameters (sigma_r & sigma_s) are pre-defined within the GPU kernel.
   This function is compatible with 32-bit and 24-bit images.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input image as a `pygame.Surface` object.
   - **kernel_size_** (*int*): The kernel size, determining the number of neighboring pixels included in the calculation.

   **Returns:**

   - **pygame.Surface**: A new 24-bit `pygame.Surface` with the bilateral filtering effect applied.

   **Raises:**

   - **TypeError**: If `surface_` is not an instance of `pygame.Surface`.
   - **ValueError**: If the surface dimensions are zero or if `kernel_size_` is negative.

   **Notes:**

   - The function extracts pixel data from `surface_` and converts it into a 3D array.
   - The `bilateral_cupy` function performs GPU-accelerated bilateral filtering and returns the processed image.

|

.. py:function:: emboss5x5_gpu(surface_)

   Apply an emboss effect to a 32-bit or 24-bit image using a 5x5 convolution kernel.

   The embossing kernel used:
       [-2.0, -1.0,  0.0,  1.0,  2.0]
       [-1.0,  1.0,  2.0,  3.0,  4.0]
       [ 0.0,  2.0,  4.0,  6.0,  8.0]
       [-1.0,  1.0,  2.0,  3.0,  4.0]
       [-2.0, -1.0,  0.0,  1.0,  2.0]

   Each RGB channel of the input image is processed independently using convolution.
   The input image must be in 32-bit or 24-bit format, and the output will be a 24-bit image.

   **Parameters:**

   - **surface_** (*pygame.Surface*): The input image as a `pygame.Surface` object.

   **Returns:**

   - **pygame.Surface**: A new 24-bit `pygame.Surface` with the emboss effect applied.

   **Raises:**

   - **TypeError**: If `surface_` is not an instance of `pygame.Surface`.
   - **ValueError**: If the surface's pixel data cannot be accessed or if its width/height is zero.

   **Notes:**

   - The function extracts pixel data from `surface_` and converts it into a 3D array.
   - The `emboss_cupy` function performs GPU-accelerated embossing and returns the processed image.

|

.. py:function:: area24_gpu(x, y, rgb_array, mask_alpha, intensity=1.0, color=cupy.asarray([128.0, 128.0, 128.0], dtype=cupy.float32, copy=False))

   Applies a color overlay to a specified area in a 24-bit RGB image using GPU acceleration.

   This function modifies the color of a region in the `rgb_array` at coordinates `(x, y)`,
   blending it with the given `color` based on the `mask_alpha` and `intensity` parameters.
   It utilizes CuPy for fast GPU-based computation.

   **Parameters:**

   - **x** (*int*):
     X-coordinate of the target pixel in the image.

   - **y** (*int*):
     Y-coordinate of the target pixel in the image.

   - **rgb_array** (*cupy.ndarray*):
     A 3D array representing the RGB image stored in GPU memory.
     Must have a shape of `(height, width, 3)`.

   - **mask_alpha** (*cupy.ndarray*):
     Alpha mask determining the transparency effect applied to the color overlay.
     Must match the shape of `rgb_array`.

   - **intensity** (*float*, optional, default=`1.0`):
     Scaling factor for the overlay effect.
     Determines how strongly the `color` blends with the original image.
     Must be in the range `[0.0, 1.0]`.

   - **color** (*cupy.ndarray*, optional, default=`[128.0, 128.0, 128.0]`):
     A 3-element array representing the RGB color to be blended.

   **Returns:**

   - (*tuple*):
     The modified RGB values at `(x, y)` after applying the effect.

   **Raises:**

   - **ValueError**: If `x` or `y` is out of bounds.
   - **ValueError**: If `intensity` is not within `[0.0, 1.0]`.
   - **ValueError**: If `rgb_array` or `mask_alpha` have incompatible shapes.

   **Notes:**

   - The function utilizes GPU acceleration with CuPy for fast image processing.
   - The color blending effect is based on the alpha mask and intensity, allowing for customizable transparency.

|

.. py:function:: brightness_gpu(surface_, brightness_factor, grid_, block_)

   Adjusts the brightness of a given image using GPU acceleration.

   This function modifies the brightness of a `pygame.Surface` by applying a
   GPU-accelerated transformation. It efficiently processes 24-bit and 32-bit
   images, leveraging CUDA for parallel computation.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input image to be processed. Must be in 24-bit or 32-bit format.

   - **brightness_factor** (*float*):
     The brightness scaling factor.
     - Values greater than `1.0` increase brightness.
     - Values between `0.0` and `1.0` decrease brightness.
     - `1.0` keeps the original brightness.

   - **grid_** (*tuple*):
     Grid dimensions `(grid_y, grid_x)`, e.g., `(25, 25)`.
     Defines how the image is divided for parallel GPU execution.
     Must match the texture and array sizes.

   - **block_** (*tuple*):
     Block dimensions `(block_y, block_x)`, e.g., `(32, 32)`.
     Determines the number of threads per block for CUDA execution.
     The product `block_x * block_y` must not exceed `1024` (CUDA's max thread limit per block).

   **Returns:**

   - (*pygame.Surface*):
     A new surface with modified brightness in 24-bit format.

   **Raises:**

   - **TypeError**: If `surface_` is not of type `pygame.Surface`.
   - **ValueError**: If `brightness_factor` is negative.

   **Notes:**

   - The function leverages GPU acceleration to efficiently process the image,
     making it suitable for large images or real-time applications.

|

.. py:function:: hsl_gpu(surface_, val_, grid_=None, block_=None)

   Applies HSL (Hue, Saturation, Lightness) rotation to an image using GPU acceleration.

   This function rotates the color of the pixels in an image or texture represented by a `pygame.Surface`
   by modifying the hue component of the HSL color model. It supports both 24-bit and 32-bit image formats.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input image surface to be processed.

   - **val_** (*float*):
     A float representing the hue rotation value. The hue is adjusted by rotating its value in the HSL color space.

   - **grid_** (*tuple*, optional):
     Grid dimensions `(grid_y, grid_x)`, e.g., `(25, 25)`. Defines how the image is divided for parallel GPU execution.
     The grid and block dimensions must match the texture and array sizes.

   - **block_** (*tuple*, optional):
     Block dimensions `(block_y, block_x)`, e.g., `(32, 32)`. Specifies the number of threads per block for CUDA execution.
     The maximum number of threads per block is 1024, so `block_x * block_y` should not exceed this limit.

   **Returns:**

   - (*pygame.Surface*):
     A new `pygame.Surface` with the modified hue based on the given rotation value.

   **Notes:**

   - This function applies a GPU-accelerated operation to rotate the hue component of the image, enabling efficient processing for large images or real-time applications.
   - The `grid_` and `block_` parameters are optional but must be provided for parallel processing to work optimally on the GPU.

|

.. py:function:: dithering_gpu(gpu_array_, grid_, block_, factor_=1.0)

   Applies a dithering effect to an image using GPU acceleration.

   This function applies a dithering effect to an RGB image using a CUDA kernel. The operation is performed
   on the GPU for fast processing. The image is first normalized, processed using the dithering kernel,
   and then converted back to 8-bit format.

   **Parameters:**

   - **gpu_array_** (*cupy.ndarray*):
     A CuPy array of shape `(w, h, 3)` containing the image data in RGB format. Must be of dtype `cupy.uint8`.

   - **grid_** (*tuple*):
     CUDA grid dimensions `(grid_y, grid_x)`. Defines how the computation is distributed across the GPU.

   - **block_** (*tuple*):
     CUDA block dimensions `(block_y, block_x)`. The number of threads per block should not exceed 1024.

   - **factor_** (*float*, optional, default=`1.0`):
     A factor controlling the dithering intensity. A higher value increases the dithering effect.

   **Returns:**

   - (*pygame.Surface*):
     A new `pygame.Surface` containing the processed image in RGB format.

   **Raises:**

   - **TypeError**: If `gpu_array_` is not a CuPy ndarray.
   - **ValueError**: If `gpu_array_` is not of dtype `cupy.uint8`.

   **Notes:**

   - The function uses a CUDA kernel (`dithering_kernel`) for high-performance processing.
   - The input image is normalized to [0, 1] before processing and then converted back to [0, 255].
   - The grid and block sizes must be chosen appropriately based on the image dimensions and GPU resources.

|

.. py:function:: fisheye_gpu(surface_, focal, focal_texture, grid_, block_)

   Applies a fisheye lens effect to a pygame.Surface using GPU acceleration.

   This function applies a fisheye distortion effect to a given surface using CUDA-based processing.
   It is optimized for real-time rendering and can be used to display game scenes through a lens effect.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     Input pygame surface in 24-bit or 32-bit format.

   - **focal** (*float*):
     Focal length of the fisheye effect, controlling the strength of the distortion.

   - **focal_texture** (*float*):
     Texture focal length, further adjusting the fisheye intensity.

   - **grid_** (*tuple*):
     CUDA grid dimensions `(grid_y, grid_x)`. Defines how computation is distributed across the GPU.

   - **block_** (*tuple*):
     CUDA block dimensions `(block_y, block_x)`. The maximum number of threads per block should not exceed 1024.

   **Returns:**

   - (*pygame.Surface*):
     A new `pygame.Surface` with the applied fisheye lens effect.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface` instance.
   - **ValueError**: If the function cannot access the pixel data of `surface_`.

   **Notes:**

   - This function leverages CUDA for high-performance image processing.
   - The effect is applied directly to the input surface and returned as a transformed surface.
   - Ensure that the grid and block values are chosen to match the texture and array sizes for optimal performance.

|

.. py:function:: swirl_gpu(surface_, rad, grid_, block_, centre_x, centre_y)

   Applies a swirl distortion effect to an image using GPU acceleration.

   This function creates a swirl effect on a given `pygame.Surface` by applying a CUDA-based
   transformation. The effect distorts pixels radially around a specified center point.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     Input surface, must be in 24-bit or 32-bit format.

   - **rad** (*float*):
     Rotation angle in radians, controlling the intensity of the swirl.

   - **grid_** (*tuple*):
     CUDA grid size, typically `(grid_y, grid_x)`. Defines how computation is distributed across the GPU.

   - **block_** (*tuple*):
     CUDA block size, typically `(block_y, block_x)`. The number of threads per block should not exceed 1024.

   - **centre_x** (*int*):
     X-coordinate of the swirl center (must be greater than 0).

   - **centre_y** (*int*):
     Y-coordinate of the swirl center (must be greater than 0).

   **Returns:**

   - (*pygame.Surface*):
     A new `pygame.Surface` with the applied swirl effect.

   **Raises:**

   - **ValueError**: If the input surface cannot be referenced as a 3D array.
   - **TypeError**: If `surface_` is not a valid `pygame.Surface`.

   **Notes:**

   - The function operates directly on the GPU using CuPy for high performance.
   - Ensure `grid_` and `block_` are correctly configured for optimal execution.
   - The function calls `swirl_cupy`, which handles the actual GPU-based processing.

|

.. py:function:: wave_gpu(surface_, rad_, size_, grid_, block_)

   Creates a wave effect on an image using GPU acceleration.

   This function applies a wave distortion effect to a given `pygame.Surface` by applying a CUDA-based
   transformation. The effect creates a realistic wave motion across the image based on the specified
   angle and block size.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input image surface, which must be in 24-bit or 32-bit format.

   - **rad_** (*float*):
     A variable angle in radians that controls the wave effect.

   - **size_** (*int*):
     The block size for the wave effect. For a realistic wave, it's recommended to keep the size below 15.

   - **grid_** (*tuple*):
     CUDA grid size, typically `(grid_y, grid_x)`, defines how computation is distributed across the GPU.

   - **block_** (*tuple*):
     CUDA block size, typically `(block_y, block_x)`, the number of threads per block should not exceed 1024.

   **Returns:**

   - (*pygame.Surface*):
     A new `pygame.Surface` with the wave effect applied. It is recommended to rescale the image to hide any visible
     distortions around the left and bottom borders, especially if a texture wraparound effect is noticeable.

   **Notes:**

   - Example usage:
     `IMAGE = wave_gpu(IMAGE, 8 * math.pi/180.0 + FRAME/10, 8, grid, block)`
     `IMAGE = scale(IMAGE, (WIDTH + 16, HEIGHT + 16))` to hide the wraparound effect.

   - The function operates directly on the GPU using CuPy for high performance.
   - Ensure that the grid and block configurations are correctly selected for optimal performance.

|

.. py:function:: chromatic_gpu(surface_, delta_x, delta_y, grid_, block_, zoom=0.999, fx=0.05)

   Applies a chromatic displacement effect to a given surface using GPU acceleration.

   This effect creates a color separation by shifting the RGB channels based on the specified
   horizontal and vertical displacement values. The function is optimized for 24-bit and 32-bit image formats.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input surface to be processed. Must be in 24-bit or 32-bit format.

   - **delta_x** (*int*):
     Horizontal displacement for the chromatic shift. Affects the red and blue channels, creating a color separation effect.
     A value of zero results in no horizontal displacement (if `fx` is low).

   - **delta_y** (*int*):
     Vertical displacement for the chromatic shift. Functions similarly to `delta_x`, but applies to vertical movement.
     A value of zero results in no vertical displacement (if `fx` is low).

   - **grid_** (*tuple*):
     A tuple `(grid_y, grid_x)` representing grid dimensions, e.g., `(25, 25)`. Defines the division of the image for GPU processing, and
     must match the texture and array sizes.

   - **block_** (*tuple*):
     A tuple `(block_y, block_x)` representing block dimensions, e.g., `(32, 32)`. Determines the number of threads per block in GPU processing.
     The product `block_x * block_y` must not exceed 1024 (the CUDA limit for threads per block).

   - **zoom** (*float*, optional, default=`0.999`):
     A zoom factor controlling the image scale after processing. Must be in the range `(0.0, 1.0)`. Default is `1.0` (no zoom).

   - **fx** (*float*, optional, default=`0.05`):
     Intensity of the chromatic effect, controlling the color separation strength. Must be within the range `[0.0, 0.2]`. Default is `0.05`.

   **Returns:**

   - (*cupy.ndarray*):
     A CuPy array representing the processed image with the chromatic displacement effect applied.
     The output format is a 24-bit image.

   **Raises:**

   - **TypeError**: If `surface_` is not of type `pygame.Surface`.
   - **ValueError**: If `surface_` has zero width or height.
   - **ValueError**: If `delta_x` or `delta_y` is negative.

   **Notes:**

   - The function leverages GPU acceleration for efficient chromatic displacement.
   - Ensure that the grid and block sizes are chosen appropriately for the image size and GPU capabilities.
   - The `fx` parameter controls the strength of the chromatic separation effect,
     and `zoom` allows for resizing after the transformation.

|

.. py:function:: rgb_split_gpu(surface_, delta_x, delta_y, grid_, block_)

   Apply an RGB split effect to an image using GPU acceleration.

   This function shifts the red, green, and blue channels of an image by `delta_x` and `delta_y`
   to create a chromatic aberration (glitch) effect. The transformation is performed entirely on
   the GPU using CuPy.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input image in a 24-bit or 32-bit format.

   - **delta_x** (*float*):
     The horizontal shift applied to the RGB channels. A positive value shifts the channels to the right,
     and a negative value shifts them to the left.

   - **delta_y** (*float*):
     The vertical shift applied to the RGB channels. A positive value shifts the channels down,
     and a negative value shifts them up.

   - **grid_** (*tuple*):
     A tuple `(grid_y, grid_x)` representing the grid size for CUDA execution, defining the distribution of computation across the GPU.

   - **block_** (*tuple*):
     A tuple `(block_y, block_x)` representing the block size, which determines the number of threads per block in GPU execution.

   **Returns:**

   - (*pygame.Surface*):
     A new surface with the RGB split effect applied.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If `delta_x` or `delta_y` are non-positive values or if the function cannot access the pixel data from `surface_`.

   **Notes:**

   - The RGB split effect creates a chromatic distortion, commonly used in glitch effects.
   - `delta_x` and `delta_y` define how far the color channels are shifted.
   - Ensure `grid_` and `block_` values are optimized for GPU execution.

|

.. py:function:: zoom_gpu(surface_, centre_x, centre_y, grid_, block_, zoom=1.0)

   Applies a zoom effect using GPU acceleration.

   This function processes a `pygame.Surface` by mapping its pixel data to GPU memory
   and applying a zoom transformation. It utilizes CUDA acceleration for high-performance image processing.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input surface to be processed. Must be a 24-bit or 32-bit format image.

   - **centre_x** (*unsigned int*):
     The x-coordinate of the zoom center. Must be greater than or equal to 0.

   - **centre_y** (*unsigned int*):
     The y-coordinate of the zoom center. Must be greater than or equal to 0.

   - **grid_** (*tuple*):
     A tuple `(grid_y, grid_x)` representing the grid dimensions. Defines the division of the image for parallel GPU processing.
     The values must match the texture and array sizes.

   - **block_** (*tuple*):
     A tuple `(block_y, block_x)` representing the block dimensions. Determines the number of threads per block in GPU execution.
     The product `block_x * block_y` must not exceed `1024` (CUDA limit).

   - **zoom** (*float*, optional):
     A zoom factor controlling the scaling of the image. Must be within the range `(0.0, 1.0]`. Default is `1.0` (no scaling).

   **Returns:**

   - (*CuPy ndarray*):
     A CuPy array containing the transformed image with zoom applied, in 24-bit format.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**:
     - If the width or height of `surface_` is 0.
     - If `zoom` is not within the range `(0.0, 1.0]`.
     - If `centre_x` or `centre_y` is negative.
     - If `surface_` cannot be mapped to a 3D array.

   **Notes:**

   - This function performs the zoom effect by mapping pixel data to GPU memory and using CUDA for high-performance processing.
   - Ensure `grid_` and `block_` are optimized for your GPU's capabilities and the image size.

|

.. py:function:: wavelength_map_gpu(surface_, grid_, block_, layer_=0)

   Apply a channel-specific color effect (redscale, greenscale, or bluescale) to an image.

   This function applies a redscale, greenscale, or bluescale effect to a given `pygame.Surface` image.
   It isolates and emphasizes one of the color channels (Red, Green, or Blue) depending on the provided
   `layer_` parameter. The operation is performed using GPU acceleration for better performance.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input image to which the effect will be applied.

   - **grid_** (*tuple*):
     A tuple `(grid_y, grid_x)` representing the grid configuration. The grid values should match the image
     and array sizes.

   - **block_** (*tuple*):
     A tuple `(block_y, block_x)` representing the block configuration. The block values determine how many threads
     are launched per block.

   - **layer_** (*unsigned short int*, optional):
     Specifies which color channel to isolate. The default is `0` (Red channel). The values are:
     - `0` for Red channel
     - `1` for Green channel
     - `2` for Blue channel

   **Returns:**

   - (*pygame.Surface*):
     The image with the selected channel effect (redscale, greenscale, or bluescale) applied.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If the `layer_` value is not within the range of 0 to 2.

   **Notes:**

   - This function uses GPU acceleration for high-performance image processing.
   - The `layer_` parameter determines which color channel is emphasized.
   - The result is a new `pygame.Surface` with the selected color channel effect applied.

|

.. py:function:: heatmap_gpu(surface_, grid_, block_, invert_=False)

   Apply a heatmap effect to an image with optional inversion.

   This function applies a heatmap effect to an image using GPU acceleration. The heatmap effect is typically
   used to visualize intensity or magnitude across a 2D surface. The effect can be inverted based on the `invert_`
   parameter, changing the color representation.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input image to which the heatmap effect will be applied.

   - **grid_** (*tuple*):
     A tuple `(grid_y, grid_x)` representing the grid configuration. The grid values should match the texture
     and array sizes for parallel execution on the GPU.

   - **block_** (*tuple*):
     A tuple `(block_y, block_x)` representing the block configuration. It defines the number of threads per block
     for GPU execution. The product `block_x * block_y` must not exceed `1024` (CUDA's maximum threads per block).

   - **invert_** (*boolean*, optional):
     If `True`, the color range of the heatmap is reversed (inverted). The default is `False`, meaning no inversion.

   **Returns:**

   - (*pygame.Surface*):
     The image with the heatmap effect applied.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If `invert_` is not a boolean.

   **Notes:**

   - The function uses GPU acceleration for high-performance image processing.
   - The inversion of the heatmap allows for a different color representation.

|

.. py:function:: heatmap_gpu_inplace(surface_, grid_, block_, invert_=False)

   Apply heatmap effect in place to a pygame.Surface.

   This function applies a heatmap effect to the input surface in place, modifying the pixel data directly
   without returning a new surface. The heatmap visualization enhances the intensity of the image based on a color
   mapping, with the option to invert the color effect.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The surface on which the heatmap effect will be applied. The surface must be a valid `pygame.Surface` containing
     pixel data in a 3D array (RGB channels).

   - **grid_** (*tuple*):
     A tuple `(grid_y, grid_x)` representing the grid configuration for parallel GPU execution. The grid values must
     match the array and texture sizes.

   - **block_** (*tuple*):
     A tuple `(block_y, block_x)` representing the block configuration. It defines the number of threads per block
     for GPU computation. The product `block_x * block_y` must not exceed `1024` (CUDA's maximum threads per block).

   - **invert_** (*boolean*, optional):
     If `True`, the heatmap effect is inverted, reversing the color range. Default is `False` (no inversion).

   **Returns:**

   - (*void*):
     This function modifies the `surface_` directly and does not return a value.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If `invert_` is not a boolean.

   **Notes:**

   - The function uses GPU acceleration for high-performance image processing.
   - The effect is applied in place, meaning the input `surface_` is modified directly.
   - The inversion of the heatmap allows for a reversed color representation.

|

.. py:function:: predator_gpu(surface_, grid_, block_, bloom_smooth=50, bloom_threshold=50, inv_colormap=False, blend=pygame.BLEND_RGB_ADD, bloom_flag=pygame.BLEND_RGB_ADD)

   Apply predator vision effect using GPU acceleration.

   This function processes the input `pygame.Surface` to apply a predator vision effect using GPU acceleration. The
   effect simulates a vision filter with bloom and colormap adjustments, which can be customized based on the
   provided parameters.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input surface to process.

   - **grid_** (*tuple*):
     A tuple `(grid_y, grid_x)` representing the grid dimensions for CUDA kernel execution.

   - **block_** (*tuple*):
     A tuple `(block_y, block_x)` representing the block dimensions for CUDA kernel execution.

   - **bloom_smooth** (*unsigned int*, optional):
     The smoothing factor for the bloom effect. Default is `50`.

   - **bloom_threshold** (*unsigned int*, optional):
     The intensity threshold for the bloom effect. Default is `50`.

   - **inv_colormap** (*boolean*, optional):
     Whether to invert the colormap. Default is `False`.

   - **blend** (*int*, optional):
     The blending mode for final rendering. Default is `pygame.BLEND_RGB_ADD`.

   - **bloom_flag** (*boolean*, optional):
     Flag to enable the bloom effect. Default is `pygame.BLEND_RGB_ADD`.

   **Returns:**

   - (*pygame.Surface*):
     The processed surface with the applied predator vision effect.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If `bloom_smooth`, `bloom_threshold` are not positive integers or if `blend` is not a valid blend mode.

   **Notes:**

   - This function uses GPU acceleration to apply a fast predator vision effect.
   - The bloom effect simulates glowing areas of high intensity, and the colormap can be inverted for different visual effects.
   - The `blend` and `bloom_flag` parameters allow for customization of the final rendering effect.

|

.. py:function:: downscale_surface_gpu(surface_, grid_, block_, zoom, w2, h2)

   Downscale a pygame surface using CUDA on CuPy.

   This function extracts pixel data from a `pygame.Surface`, transfers it to the GPU, applies a
   downscaling operation using a CUDA kernel, and returns the processed image in a byte buffer.

   **Parameters:**

   - **surface_** (*pygame.Surface*):
     The input surface to be downscaled.

   - **grid_** (*tuple*):
     CUDA grid dimensions for kernel execution.

   - **block_** (*tuple*):
     CUDA block dimensions for kernel execution.

   - **zoom** (*float*):
     The scaling factor for downscaling. Must be in the range `[0.0, 0.9999]`.

   - **w2** (*int*):
     The target width of the downscaled image.

   - **h2** (*int*):
     The target height of the downscaled image.

   **Returns:**

   - (*bytes*):
     The downscaled image in RGB format as a byte buffer.

   **Raises:**

   - **TypeError**: If `surface_` is not a `pygame.Surface`.
   - **ValueError**: If `zoom` is not in the range `[0.0, 0.9999]`, or if `w2` or `h2` are non-positive values.

   **Notes:**

   - This function uses GPU acceleration with CuPy for efficient downscaling.
   - The downscaled image is returned as a byte buffer in RGB format.
   - Ensure that the grid and block values are optimized for GPU performance.

|




