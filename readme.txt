LcR Super Resolution on Nvidia Jetson TK1 in C/C++
==================================================

Description
===========
The program uses Locality-constrained Representation (LcR) algorithm[1] to super-resolve images from 15x15 pixels to 60x60 pixels. It has been optimized using Nvidia CUDA on the Jetson TK1 but can run on any CUDA-capable GPU of compute capability 2.0 or above (cuBLAS library requirement). 

1: Junjun Jiang, Ruimin Hu, Zhongyuan Wang, Zhen Han: Noise Robust Face Hallucination via Locality-Constrained Representation. IEEE Transactions on Multimedia 16(5): 1268-1281 (2014)

Configuration
=============
i) Arguments: The program takes 6 input arguments mentioned in order
	1) Input image.
	2) Number of images to use to super resolve: Out of the database only the image patches closest to the low resolution patches are utilized. This argument specifies how many of those closest images should be used.
	3) Patch Size: Size of each patch in pixels to super resolve.
	4) Overlap: Overlap of pixels for each patch. Larger overlap means less pixel distance between the two patches but greater number of patches to solve for.
	5) Regularization parameter: As mentioned in the paper.
	6) Detect face: Y/y to use face detection, anything else to bypass detection.
	
	argument example: "./program testimage.jpg 200 6 4 0.3 y"
	Input face = testimage.jpg. 200 images will be used to solve the face. Patch size of 6 pixels with overlap of 4 pixels. Regularization parameter = 0.3. Facial detection is carried out.

ii) Inputs/Directories:
	1) Training Database: The training database consists of low resolution 15x15 pixel images and their high resolutions versions at 60x60 pixels. Each LR image should have a corresponding HR image with the same filename. The following parameters have to be defined:
		a) LR_DIR: Set using #define in code, directory of LR training images.
		b) HR_DIR: Set using #define in code, directory of HR training images.
		c) input.txt: In the same directory as the program. The file contains the number of training images on the first line followed by the filenames of all images.
		
		input.txt example:
			100
			face001.pgm
			face002.pgm
			...
		for 100 images with face001.pgm and face002.pgm as the first two images.

	2) HaarCascade directory: CASCADE_DIR set using #define in code, points to directory with the haarcascade being used.
	3) Output file directory and name: SAVE_DIR set using #define in code.
	
iii) Directory Structure:
		input.txt				(file containing number and names of training database images)
		program					(executable)
		haarcascade_frontalface_default.xml	(default OpenCV facial detection haarcascade)
		makefile
		testimage.jpg				(input file used for testing)
		testprog.sh				(sample script to super-resolve multiple images)
		HR/					(60x60 pixel training images)
			Face000.pgm
			Face000M.pgm			(mirrored version of Face000.pgm)
			Face001.pgm
			Face001M.pgm
			Face002.pgm
			...
		LR/					(15x15 pixel training images)
			Face000.pgm
			Face000M.pgm
			Face001.pgm
			Face001M.pgm
			Face002.pgm
			...
		Full Size Database/			(221x221 pixel training images)
			Face000.pgm
			Face000M.pgm
			Face001.pgm
			Face001M.pgm
			Face002.pgm
			...
		Results/				(results will be output here)
		src/
			sr.cu				(source code)
			
iv) Compilation:
	A makefile is included for compilation. OpenCV and CUDA toolkit need to be installed.

v) Libraries used:
	1) OpenCV (http://opencv.org/): Used for Computer Vision applications. The modules opencv_core, opencv_highgui, opencv_imgproc and opencv_objdetect are used in this program. A version for OpenCV optimized for Tegra by Nvidia OpenCV4Tegra 2.4.8.2 is used.
	2) cuBLAS (https://developer.nvidia.com/cublas): Used for linear algebraic operations on Nvidia GPU. Nvidia CUDA toolkit 6.0 for Jetson TK1 is used which includes the cuBLAS library.
	Both libraries are available for download from Jetson TK1 support site: https://developer.nvidia.com/jetson-tk1-support
	
Psuedocode
==========
-Allocate and initialize host and device memory
-Load training image database by reading from input.txt file
-Calculate "pixel weight", the number of overlapping patches on each pixel
-Load input image and detect face if required
-Resize input image to 15x15 pixels
-For each patch:
	-Find the closest (by euclidean norm) patches
	-Find the difference between the rasterized input patch and training patches to produce difference matrix (patch size^2 x number of images)
	-Find the diagonal euclidean norm matrix (number of images x number of images)
	-Patch input matrix = difference matrix * transpose(difference matrix) + regularization parameter * euclidean norm matrix
-Batched factorization of all matrices
-Batched inversion of all matrices
-For each patch:
	-Multiply inverse by ones vector to find weights (solves A*x=B where A=input matrix, x=weights, B=ones vector)
	-Find the sum of weights for normalization
	-Create HR image, for each LR pixel:
		-Fill 4 (magnification factor) HR pixels
		-HR pixel += normalized weight * HR training image / number of patches overlapping pixel
-Save HR face image
