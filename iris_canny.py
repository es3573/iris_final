#!/usr/bin/env python
#from segmentation import *
from tabulate import tabulate
import time
import math
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from pycuda import driver, compiler, gpuarray, tools
print "Iris Detection Algorithm: Canny Edge Detection"

#Device Source
mod = SourceModule("""
#include <math.h>
#define PI 3.14159265

	__global__ void grayscale(float* data_r, float* data_g, float* data_b, float* grayscaled, int numCols){
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int row = blockIdx.y*blockDim.y + ty;
		int col = blockIdx.x*blockDim.x + tx;
		
		int index = row*numCols + col;
		
		grayscaled[index] = 0.2989*data_r[index] + 0.5870*data_g[index] + 0.1140*data_b[index];
		
	}
	
	__global__ void convolution_2D_tiled(float* data, float* mask, float* z, int M, int N, int dilation, int mask_size){
	
	//THIS CODE IS FOR AN UNPADDED INPUT, JUST USE PADDED INPUT AKA CONVERT MY PYTHON CODE INTO KERNEL SHIT
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int row = blockIdx.y*blockDim.y + ty;
		int col = blockIdx.x*blockDim.x + tx;
		
		//this works for smaller matrices (input matrix < 20000 elements)
		int output = 0;
		if ((row >= dilation) && (row < dilation + M) && (col >= dilation) && (col < dilation + N)){
			for (int j = 0; j < mask_size; j++){
				for (int k = 0; k < mask_size; k++){
					output += mask[j*mask_size +k] * data[(row-dilation + dilation*j)*(N+dilation*(mask_size - 1)) + (col-dilation + dilation*k)];
				}
			}
			z[(row - dilation)*N + (col - dilation)] = output;
		}
	}

	__global__ void gradient(float* data_x, float* data_y, float* gradient, int numCols){
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int row = blockIdx.y*blockDim.y + ty;
		int col = blockIdx.x*blockDim.x + tx;
		
		int index = row*numCols + col;
		
		gradient[index] = sqrt(data_x[index]*data_x[index] + data_y[index]*data_y[index]);
		
	}
	
	__global__ void orientation(float* data_x, float* data_y, float* orientation, int numCols){
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int row = blockIdx.y*blockDim.y + ty;
		int col = blockIdx.x*blockDim.x + tx;
		
		int index = row*numCols + col;
		
		float value = atan2(-data_y[index],data_x[index]);
		if (value < 0){
			value = value + PI;
		}
		orientation[index] =  value * 180 / PI;	
	}
	
	__global__ void adjgamma(float* gradient_in, float* gradient_out, float min, float max, int numCols){
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int row = blockIdx.y*blockDim.y + ty;
		int col = blockIdx.x*blockDim.x + tx;
		
		int index = row*numCols + col;
		float gamma = 1/1.9;
		
		float value = (gradient_in[index] - min) / max;
		gradient_out[index] = pow(value, gamma);

	}
	
	__global__ void nonmaxsuppression(float* im, float* inimage, int* orient, float* xoff, float* yoff, float* hfrac, float* vfrac, int iradius, int numCols, int numRows){
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int row = blockIdx.y*blockDim.y + ty + iradius;
		int col = blockIdx.x*blockDim.x + tx + iradius;
		
		if ((row < numCols) && (col < numCols)){
			int index = row*numCols + col;
			int ori = orient[index];
			
			double x = col + xoff[ori];
			double y = row -yoff[ori];
			
			int fx = (int) x;
			int cx = (int)(x+0.5);
			int fy = (int) y;
			int cy = (int)(y+0.5);
			float tl = inimage[fy*numCols + fx];
			float tr = inimage[fy*numCols + cx];
			float bl = inimage[cy*numCols + fx];
			float br = inimage[cy*numCols + cx];
			
			float upperavg = tl + hfrac[ori]*(tr-tl);
			float loweravg = bl + hfrac[ori]*(br-bl);
			float v1 = upperavg + vfrac[ori]*(loweravg - upperavg);
			
			if (inimage[index] > v1){
				x = col - xoff[ori];
				y = row + yoff[ori];
				fx = (int)(x);
				cx = (int)(x+0.5);
				fy = (int)(y);
				cy = (int)(y+0.5);
				tl = inimage[fy*numCols + fx];
				tr = inimage[fy*numCols + cx];
				bl = inimage[cy*numCols + fx];
				br = inimage[cy*numCols + cx];
			
				upperavg = tl + hfrac[ori]*(tr-tl);
				loweravg = bl + hfrac[ori]*(br-bl);
				float v2 = upperavg + vfrac[ori]*(loweravg - upperavg);
				
				if (inimage[index] > v2){
					im[index] = inimage[index];
				}
				
			}
			
		}
	}		
		
	
	
""")

# create_canny() takes the eye image file and all the timing vectors and outputs the numpy arrays neccessary to display in MATLAB plots
# 	The Canny Edge Detection Method can be broken down into steps: grayscaling, gaussian smooth filtering, intensity and orientation gradient mapping, adjusting the gamma value (basically the contrast), 
#   applying non-maximum suppression, and finally a double-threshold hysterisis in order to develop an edge map. 
			
def create_canny(file_name, x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time):
	#function calls to port pyCUDA kernels 	
	grayscaled = mod.get_function("grayscale") 
	conv = mod.get_function("convolution_2D_tiled")
	grad = mod.get_function("gradient")
	orient = mod.get_function("orientation")
	adjg = mod.get_function("adjgamma")
	nonmaxsuppression = mod.get_function("nonmaxsuppression")
	
	###################### Grayscaling ######################
	# Grayscaling is used in order to commit to 2D arrays typical of image processing techniques/functions that follow
	# Both the python and pyCUDA implement the same method of grayscaling which takes the 2D arrays of the RGB image and 
	# applies a scaling factor that follows by summation into a final 2D array. The 2D splicing of the RGB image is taken
	# outside of the timing loops as this is not the operation we are interested in comparing
	
	img = Image.open(file_name)
	img.load()
	data_raw = np.asarray(img, dtype="float32")
	numCols = data_raw.shape[1]
	numRows = data_raw.shape[0]
	r, g, b = data_raw[:,:,0], data_raw[:,:,1], data_raw[:,:,2]
	print(data_raw.shape)
	
	# python RGB 
	times = []
	for w in range(1,4):
		start = time.time()
		gray = 0.2989*r + 0.5870*g + 0.1140*b
		times.append(time.time() - start)
	print 'python grayscaling time: ', np.average(times)
	py_gray_time.append(np.average(times))
	
	gray = np.floor(gray)

	M = gray.shape[0]
	N = gray.shape[1]
	x_size.append(M*N)

	r_gpu = gpuarray.to_gpu(r)	
	g_gpu = gpuarray.to_gpu(g)
	b_gpu = gpuarray.to_gpu(b)
	grayscaled_gpu = gpuarray.empty(gray.shape, np.float32)

	# pyCUDA RGB
	for w in range(1,4):
		start = time.time()
		grayscaled(r_gpu, g_gpu, b_gpu, grayscaled_gpu, np.int32(numCols), block=(8, 8, 1), grid = (N/8 + 1, M/8 + 1, 1))
		times.append(time.time() - start)
	print 'pyCUDA grayscaling time: ', np.average(times)
	pyCUDA_gray_time.append(np.average(times))

	####################### Smoothing Gaussian Filter ######################
	# Gaussian smoothing is neccessary in the Canny Edge Detection method  as the raw image has inherently a lot of edges  that
	# can be misperceived as true edges in the final edge map development. Smoothing would effectively blur the image in a way
	# that the following algorithms would better detect "true" edges, which we define as edges more easily discerned by the human
	# eye. Both python and pyCUDA versions implement the same algorithm which is a basic 2D convolution of the Gaussian mask filter
	# and the grayscaled image.
	
	# The first step is not considered in the actual algorithm timings as it is simply creating the gaussian filter to be used in the 
	# convolution. The Gaussian mask itself was taken from MATLAB's fspecial() function with a sigma value of 2, size of 13x13. Due to the 
	# symmetry of Gaussian filters, there is no need to rotate the mask when applying 2D convolution. We then used cv2.filter2D() function 
	# from the cv2 package in ordder to compare with our pyCUDA implementation.
	
	dilation = 1
	mask_size = 13
	sigma = 2
	padding = ((mask_size - 1)/2)*dilation
	ind = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
	xmask = []
	for i in range(13):
		xmask.append(ind)
	xmask_np = np.array(xmask, dtype = 'float32')
	ymask_np = np.transpose(xmask_np)
	h = np.exp(-(xmask_np**2 + ymask_np**2)/(2*sigma*sigma))
	h = h/np.sum(h)


	data = np.pad(grayscaled_gpu.get().astype(np.float32), (int(padding),int(padding)), 'constant', constant_values = 0)
	data_gpu = gpuarray.to_gpu(data)
	mask = h.astype(np.float32)
	mask_gpu = gpuarray.to_gpu(mask)
	z_gpu = gpuarray.empty(gray.shape, np.float32)

	# python 2D convolution
	times = []
	for q in range(1,4):
		start = time.time()
		dst = cv2.filter2D(gray,-1,mask)
		times.append(time.time() - start)
	print 'python filter time: ', np.average(times)
	py_filter_time.append(np.average(times))

	# pyCUDA 2d convolution
	times = []
	for q in range(1,4):
		start = time.time()
		conv(data_gpu, mask_gpu, z_gpu, np.int32(M), np.int32(N), np.int32(dilation), np.int32(mask_size), block=(8, 8, 1), grid = (N/8 + 1, M/8 + 1, 1))
		times.append(time.time() - start)
	print 'pyCUDA filter time: ', np.average(times)
	pyCUDA_filter_time.append(np.average(times))

	###################### Intensity Gradient Mapping ######################
	# The step of intensity gradient mapping can be simply reduced to creating a "texture" to the smoothed image. The "texture" of an image is analogous to
	# a topological map of a mountain range. The steeper the slope of the mountain, the greater it's intensity map. Likewise, the steeper the slope of the 
	# differentials, the greater the intensity. This will correlate with stronger edges perceived. This step requires a preprocessing step which parses the 
	# image array into different subregions (top-right, top-left, bottom-right, bottom-left) in order to precalculate arrays differentials. The actual 
	# algorithm following this preprocessing step is the basic Euclidean distance which we compare between python and pyCUDA. Both python and pyCUDA implementations
	# compare this basic equation of Euclidean distance.
	
	h1 = np.c_[z_gpu.get()[:,1:numCols], np.zeros(numRows)]
	h2 = np.c_[np.zeros(numRows), z_gpu.get()[:,0:numCols-1]]
	h = h1 - h2

	v1 = np.r_[z_gpu.get()[1:numRows,:], [np.zeros(numCols)]]
	v2 = np.r_[[np.zeros(numCols)], z_gpu.get()[0:numRows-1, :]]
	v = v1 - v2

	d1a = np.c_[z_gpu.get()[1:numRows,1:numCols], np.zeros(numRows-1)] #add col
	d1b = np.pad(z_gpu.get()[1:numRows, 1:numCols], (1,1), 'constant', constant_values = 0)
	d1b = d1b[1:numRows+1, 1:numCols+1]
	d1c = np.pad(z_gpu.get()[0:numRows-1, 0:numCols-1], (1,1), 'constant', constant_values = 0)
	d1c = d1c[0:numRows, 0:numCols]
	d1 = d1b - d1c

	d2a = np.pad(z_gpu.get()[0:numRows-1, 1:numCols], (1,1), 'constant', constant_values = 0)
	d2a = d2a[0:numRows, 1:numCols+1]
	d2b = np.pad(z_gpu.get()[1:numRows, 0:numCols-1], (1,1), 'constant', constant_values = 0)
	d2b = d2b[1:numRows+1, 0:numCols]
	d2 = d2a-d2b

	X = (h + (d1+d2)/2.0)
	Y = (v + (d1-d2)/2.0)
	
	# python intensity gradient mapping
	times = []
	for g in range(1,4):
		start = time.time()
		gradient = np.sqrt(np.multiply(X,X) + np.multiply(Y,Y))
		times.append(time.time() - start)
	print 'python gradient time: ', np.average(times)
	py_gradient_time.append(np.average(times))

	X_gpu = gpuarray.to_gpu(X.astype(np.float32))
	Y_gpu = gpuarray.to_gpu(Y.astype(np.float32))
	gradient_gpu = gpuarray.empty(X_gpu.shape, np.float32)

	# pyCUDA intensity gradient mapping
	times = []
	for g in range(1,4):
		start = time.time()
		grad(X_gpu, Y_gpu, gradient_gpu, np.int32(numCols), block=(8, 8, 1), grid = (N/8 + 1, M/8 + 1, 1))
		times.append(time.time() - start)
	print 'pyCUDA gradient time: ', np.average(times)
	pyCUDA_gradient_time.append(np.average(times))

	####################### Orientation of the Gradients ######################
	# Since the gradient of an image also has a direction, we must also keep track of the direction of the intensity values.
	# The basic algorithm for this is as follows: taking the arctan() of the negative Y-direction differential and the X-direction
	# differential. Then it converts it to degrees. Both python and pyCUDA implement this method.
	
	# python orientation calculation
	for g in range(1,4):
		start = time.time()
		orientation = np.arctan2(-Y,X) #confirmed same as matlab
		for i in range(orientation.shape[0]):
			for j in range(orientation.shape[1]):
				if orientation[i,j] < 0:
					orientation[i,j]  = orientation[i,j] + math.pi

		orientation = orientation * 180 / math.pi
		times.append(time.time() - start)
	print 'python orientation time: ', np.average(times)
	py_orientation_time.append(np.average(times))

	X_gpu = gpuarray.to_gpu(X.astype(np.float32))
	Y_gpu = gpuarray.to_gpu(Y.astype(np.float32))
	orientation_gpu = gpuarray.empty(X_gpu.shape, np.float32)

	# pyCUDA orientation calculation
	times = []
	for g in range(1,4):
		start = time.time()
		orient(X_gpu, Y_gpu, orientation_gpu, np.int32(numCols), block = (8,8,1), grid = (N/8 + 1, M/8 + 1, 1))
		times.append(time.time() - start)
	print 'pyCUDA orientation time: ', np.average(times)
	pyCUDA_orientation_time.append(np.average(times))

	####################### Adjusting Gamma (Increasing Contrast) ######################
	# This step is taken onto the gradient mapping from the previous step and it aims to increase the contrast between the gradients.
	# Effectively, this would allow stronger edges to increase in intensity and weaker edges to be more likely to be suppressed. The basic
	# algorithm for this step is to find the min() and max() of the entire array, apply a sort of normalization, and then raise each pixel
	# by a power of (1/1.9) where 1.9 is the gamma value chosen for this experiment. Both python and pyCUDA implement this method, though we
	# decided to use pyCUDA gpuarray's built-in function for min() and max() instead of creating our own naive reduction kernel. 
	newim = gradient
	
	# python adjusting gamma implementation
	times = []
	for g in range(1,4):
		start = time.time()
		newim = newim - np.min(np.min(newim))
		newim = newim / np.max(np.max(newim))
		newim = newim ** (1/1.9)
		adjgamma = newim
		times.append(time.time() - start)
	print 'python adjgamma time: ', np.average(times)
	py_adjgamma_time.append(np.average(times))
	
	# pyCUDA adjusting gamma implementation
	times = []
	for g in range(1,4):
		start = time.time()
		min = pycuda.gpuarray.min(gradient_gpu)
		max = pycuda.gpuarray.max(gradient_gpu)
		adjgamma_gpu = gpuarray.empty(X_gpu.shape, np.float32)
		adjg(gradient_gpu, adjgamma_gpu, np.float32(min.get()), np.float32(max.get()), np.int32(numCols), block = (8,8,1), grid = (N/8 + 1, M/8 + 1, 1))
		times.append(time.time() - start)
	print 'pyCUDA adjgamma time: ', np.average(times)
	pyCUDA_adjgamma_time.append(np.average(times))

	####################### Non-maximum suppression ######################
	# Following the increased contrast step, non-maximum suppression is applied to essentially filter out the weaker edges of the gradient map. Contrast applies 
	# a sort of blurring effect due to the increased contrast, so non-maximum suppression can also be viewed as thinning the edges into more discernable marks. 
	# The basic algorithm for non-maximum suppression is a pixel-wise operation where it essentially looks at a small subregion around each pixel, finds the local
	# maxima within the region, and suppresses the non-maxima pixels. Both python and pyCUDA implement the same algorithm.
	
	inimage = adjgamma_gpu.get()
	orient = orientation
	radius = 1.5
	rows,cols = inimage.shape
	im = np.zeros((rows,cols))        #Preallocate memory for output image for speed
	iradius = np.ceil(radius).astype(np.int32)

	#Precalculate x and y offsets relative to centre pixel for each orientation angle 
	angle = np.array(range(181))*np.pi/180    	# Array of angles in 1 degree increments (but in radians).
	xoff = radius*np.cos(angle)   				# x and y offset of points at specified radius and angle
	yoff = radius*np.sin(angle)   				# from each reference position.
	hfrac = xoff - np.floor(xoff) 				# Fractional offset of xoff relative to integer location
	vfrac = yoff - np.floor(yoff) 				# Fractional offset of yoff relative to integer location

	orient = np.fix(orient).astype(np.int32)


	# python non-maximum suppression
	times = []
	for w in range(1,2):
		start = time.time()
		for row in range (iradius,(rows - iradius)):
			for col in range (iradius,(cols - iradius)):
				ori = orient[row,col]   					# Index into precomputed arrays

				x = col + xoff[ori]     					# x, y location on one side of the point in question
				y = row - yoff[ori]

				fx = np.floor(x).astype(np.int32)          	# Get integer pixel locations that surround location x,y
				cx = np.ceil(x).astype(np.int32)
				fy = np.floor(y).astype(np.int32)
				cy = np.ceil(y).astype(np.int32)
				tl = inimage[fy,fx]   						# Value at top left integer pixel location.
				tr = inimage[fy,cx]    						# top right
				bl = inimage[cy,fx]   						# bottom left
				br = inimage[cy,cx]   						# bottom right

				upperavg = tl + hfrac[ori] * (tr - tl)  	# Now use bilinear interpolation to estimate value at x,y
				loweravg = bl + hfrac[ori] * (br - bl)  	
				v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

				if inimage[row, col] > v1 : 				# Check the value on the other side
					x = col - xoff[ori]     				# x, y location on the `other side' of the point in question
					y = row + yoff[ori]

					fx = np.floor(x).astype(np.int32)
					cx = np.ceil(x).astype(np.int32)
					fy = np.floor(y).astype(np.int32)
					cy = np.ceil(y).astype(np.int32)
					tl = inimage[fy,fx]   					# Value at top left integer pixel location.
					tr = inimage[fy,cx]    					# top right
					bl = inimage[cy,fx]    					# bottom left
					br = inimage[cy,cx]    					# bottom right
					upperavg = tl + hfrac[ori] * (tr - tl)
					loweravg = bl + hfrac[ori] * (br - bl)
					v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

					if inimage[row,col] > v2:            	# This is a local maximum.
						im[row, col] = inimage[row, col] 	# Record value in the output image.
		times.append(time.time() - start)
	print 'python nonmax suppression time: ', np.average(times)
	py_nonmax_time.append(np.average(times))

	im_gpu = gpuarray.to_gpu(im.astype(np.float32))	
	inimage_gpu = gpuarray.to_gpu(inimage.astype(np.float32))
	orient_gpu = gpuarray.to_gpu(orient.astype(np.int32))
	xoff_gpu = gpuarray.to_gpu(xoff.astype(np.float32))
	yoff_gpu = gpuarray.to_gpu(yoff.astype(np.float32))
	hfrac_gpu = gpuarray.to_gpu(hfrac.astype(np.float32))
	vfrac_gpu = gpuarray.to_gpu(vfrac.astype(np.float32))

	# pyCUDA non-maximum suppression
	times = []
	for w in range(1,4):
		start = time.time()
		nonmaxsuppression(im_gpu, inimage_gpu, orient_gpu, xoff_gpu, yoff_gpu, hfrac_gpu, vfrac_gpu, np.int32(iradius), np.int32(numCols), np.int32(numRows), block = (8,8,1), grid = (N/8 + 1, M/8 + 1, 1))
		times.append(time.time() - start)
	print 'pyCUDA nonmax suppression time: ', np.average(times)
	pyCUDA_nonmax_time.append(np.average(times))
		
	###################### Double-threshold hystersis ######################
	# After non-maximum suppression, the image we receive is essentially an edge map with some noisy edges mixed in there. The goal of 
	# double-threshold hystersis is to filer out these noisy edges to obtain a cleaner edge map
	
	T1 = 0.20
	T2 = 0.19
	im = im_gpu.get()
	rows, cols = im.shape    # Precompute some values for speed and convenience.
	rc = rows*cols
	rcmr = rc - rows
	rp1 = rows+1
	imx,imy = im.shape
	bw = np.reshape(im,(imx*imy))                # Make image into a column vector
	pix = np.nonzero(bw > T1)       # Find indices of all pixels with value > T1
	pix = pix[0] #tuple to array

	npix = pix.shape[0]
	stack = np.zeros(rows*cols) # Create a stack array (that should never overflow

	stack = pix        # Put all the edge points on the stack
	stp = npix                 # set stack pointer
	for k in range (npix):
		bw[pix[k]] = -1        # mark points as edges


	# Precompute an array, O, of index offset values that correspond to the eight 
	# surrounding pixels of any point. Note that the image was transformed into
	# a column vector, so if we reshape the image back to a square the indices 
	# surrounding a pixel with index, n, will be:
	#              n-rows-1   n-1   n+rows-1
	#
	#               n-rows     n     n+rows
	#                     
	#              n-rows+1   n+1   n+rows+1

	O = [-1, 1, -rows-1, -rows, -rows+1, rows-1, rows, rows+1]

	while stp != -1 :           	# While the stack is not empty
		v = stack[stp-1]         	# Pop next index off the stack
		stp = stp - 1
		
		if v > rp1 and v < rcmr:   	# Prevent us from generating illegal indices
									# Now look at surrounding pixels to see if they
									# should be pushed onto the stack to be
									# processed as well.
		   index = O+v     			# Calculate indices of points around this pixel.     
		   for l in range(8):
			ind = index[l]
			if bw[ind-1] > T2:   	# if value > T2,
				stp = stp+1  		# push index onto the stack.
				stack[stp-1] = ind
				bw[ind-1] = -1 		# mark this as an edge point

	bw = (bw == -1)            		# Finally zero out anything that was not an edge 
	bw = np.reshape(bw,(rows,cols)) # reshape the image
	
	grayscaled_transfer = grayscaled_gpu.get()
	z_gpu_transfer = z_gpu.get()
	gradient_transfer = gradient_gpu.get()
	orientation_transfer = orientation_gpu.get()
	adjgamma_transfer = adjgamma_gpu.get()
	im_transfer = im_gpu.get()
	
	return img, grayscaled_transfer, z_gpu_transfer, gradient_transfer, orientation_transfer, adjgamma_transfer, im_transfer, bw
	

# showplot() creates MATLAB plots of each step described in create_canny()	
def showplot(img, grayscaled_transfer, z_gpu_transfer, gradient_transfer, orientation_transfer, adjgamma_transfer, im_transfer, bw,i):
	plt.figure(i)
	plt.subplot(4,2,1)
	plt.imshow(img),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(4,2,2)
	plt.imshow(grayscaled_transfer),plt.title('grayscaling')
	plt.xticks([]), plt.yticks([])
	plt.subplot(4,2,3)
	plt.imshow(z_gpu_transfer), plt.title('filtered')
	plt.xticks([]), plt.yticks([])
	plt.subplot(4,2,4)
	plt.imshow(gradient_transfer), plt.title('gradient')
	plt.xticks([]), plt.yticks([])
	plt.subplot(4,2,5)
	plt.imshow(adjgamma_transfer), plt.title('adjgamma')
	plt.xticks([]), plt.yticks([])
	plt.subplot(4,2,6)
	plt.imshow(im_transfer), plt.title('non-max suppression')
	plt.xticks([]), plt.yticks([])
	plt.subplot(4,2,7)
	plt.imshow(bw), plt.title('edge map')
	plt.xticks([]), plt.yticks([])

# showtimes() creates MATLAB plots to compare times taken by both python and pyCUDA implementations
def showtimes(python_time, pyCUDA_time, x_size,i, title):
	plt.figure(9+i)
	plt.subplot(2,1,1)
	plt.title(title)
	plt.plot(x_size, python_time, '-b', label = 'python time')
	plt.plot(x_size, pyCUDA_time, '-r', label = 'pyCUDA time')
	plt.ylabel('seconds')
	plt.legend(fontsize = 10, loc = 'upper left')
	
	plt.subplot(2,1,2)
	plt.plot(x_size, pyCUDA_time, '-b', label = 'pyCUDA time')
	plt.ylabel('seconds')
	plt.xlabel('number of pixels')
	plt.legend(fontsize = 10, loc = 'upper left')

# calculate_speedups() calculates the speedups taken by both python and pyCUDA 	
def average_speedups(py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time):
	speedups = []
	
	speedups.append(np.average([x * 1.0/y for x,y in zip(py_gray_time, pyCUDA_gray_time)]))
	speedups.append(np.average([x * 1.0/y for x,y in zip(py_filter_time, pyCUDA_filter_time)]))
	speedups.append(np.average([x * 1.0/y for x,y in zip(py_gradient_time, pyCUDA_gradient_time)]))
	speedups.append(np.average([x * 1.0/y for x,y in zip(py_orientation_time, pyCUDA_orientation_time)]))
	speedups.append(np.average([x * 1.0/y for x,y in zip(py_adjgamma_time, pyCUDA_adjgamma_time)]))
	speedups.append(np.average([x * 1.0/y for x,y in zip(py_nonmax_time, pyCUDA_nonmax_time)]))

	return speedups

# PrintSpeedUp() displays the average speedups across all cases in a nice table	
def PrintSpeedUp(speedups):
    print "Speedup(Python/pyCUDA):"
    speedup = [[s for s in zip(speedups)]]
    header = ['grayscaling', 'filtering', 'gradient', 'orientation', 'adjgamma', 'nonmaxsuppression']
    print tabulate(speedup, header, tablefmt='fancy_grid').encode('utf-8')

	
py_gray_time = []
pyCUDA_gray_time = []
py_filter_time = []
pyCUDA_filter_time = []	
py_gradient_time = []
pyCUDA_gradient_time = []	
py_orientation_time = []
pyCUDA_orientation_time = []
py_adjgamma_time = []
pyCUDA_adjgamma_time = []
py_nonmax_time = []
pyCUDA_nonmax_time = []
x_size = []
	
img1, grayscaled_transfer1, z_gpu_transfer1, gradient_transfer1, orientation_transfer1, adjgamma_transfer1, im_transfer1, bw1 = create_canny('/home/es3573/eye_1.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img2, grayscaled_transfer2, z_gpu_transfer2, gradient_transfer2, orientation_transfer2, adjgamma_transfer2, im_transfer2, bw2 = create_canny('/home/es3573/eye_2.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img3, grayscaled_transfer3, z_gpu_transfer3, gradient_transfer3, orientation_transfer3, adjgamma_transfer3, im_transfer3, bw3 = create_canny('/home/es3573/eye_3.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img4, grayscaled_transfer4, z_gpu_transfer4, gradient_transfer4, orientation_transfer4, adjgamma_transfer4, im_transfer4, bw4 = create_canny('/home/es3573/eye_4.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img5, grayscaled_transfer5, z_gpu_transfer5, gradient_transfer5, orientation_transfer5, adjgamma_transfer5, im_transfer5, bw5 = create_canny('/home/es3573/eye_5.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img6, grayscaled_transfer6, z_gpu_transfer6, gradient_transfer6, orientation_transfer6, adjgamma_transfer6, im_transfer6, bw6 = create_canny('/home/es3573/eye_6.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img7, grayscaled_transfer7, z_gpu_transfer7, gradient_transfer7, orientation_transfer7, adjgamma_transfer7, im_transfer7, bw7 = create_canny('/home/es3573/eye_7.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img8, grayscaled_transfer8, z_gpu_transfer8, gradient_transfer8, orientation_transfer8, adjgamma_transfer8, im_transfer8, bw8 = create_canny('/home/es3573/eye_8.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
img9, grayscaled_transfer9, z_gpu_transfer9, gradient_transfer9, orientation_transfer9, adjgamma_transfer9, im_transfer9, bw9 = create_canny('/home/es3573/eye_9.jpg', x_size, py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)

speedups = average_speedups(py_gray_time, pyCUDA_gray_time, py_filter_time, pyCUDA_filter_time, py_gradient_time, pyCUDA_gradient_time, py_orientation_time, pyCUDA_orientation_time, py_adjgamma_time, pyCUDA_adjgamma_time, py_nonmax_time, pyCUDA_nonmax_time)
PrintSpeedUp(speedups)

showplot(img1, grayscaled_transfer1, z_gpu_transfer1, gradient_transfer1, orientation_transfer1, adjgamma_transfer1, im_transfer1, bw1,1)
showplot(img2, grayscaled_transfer2, z_gpu_transfer2, gradient_transfer2, orientation_transfer2, adjgamma_transfer2, im_transfer2, bw2,2)
showplot(img3, grayscaled_transfer3, z_gpu_transfer3, gradient_transfer3, orientation_transfer3, adjgamma_transfer3, im_transfer3, bw3,3)
showplot(img4, grayscaled_transfer4, z_gpu_transfer4, gradient_transfer4, orientation_transfer4, adjgamma_transfer4, im_transfer4, bw4,4)
showplot(img5, grayscaled_transfer5, z_gpu_transfer5, gradient_transfer5, orientation_transfer5, adjgamma_transfer5, im_transfer5, bw5,5)
showplot(img6, grayscaled_transfer6, z_gpu_transfer6, gradient_transfer6, orientation_transfer6, adjgamma_transfer6, im_transfer6, bw6,6)
showplot(img7, grayscaled_transfer7, z_gpu_transfer7, gradient_transfer7, orientation_transfer7, adjgamma_transfer7, im_transfer7, bw7,7)
showplot(img8, grayscaled_transfer8, z_gpu_transfer8, gradient_transfer8, orientation_transfer8, adjgamma_transfer8, im_transfer8, bw8,8)
showplot(img9, grayscaled_transfer9, z_gpu_transfer9, gradient_transfer9, orientation_transfer9, adjgamma_transfer9, im_transfer9, bw9,9)

showtimes(py_gray_time, pyCUDA_gray_time, x_size,1, 'grayscaling')
showtimes(py_filter_time, pyCUDA_filter_time, x_size,2, 'filtered')
showtimes(py_gradient_time, pyCUDA_gradient_time, x_size,3, 'gradient')
showtimes(py_orientation_time, pyCUDA_orientation_time, x_size,4, 'orientation')
showtimes(py_adjgamma_time, pyCUDA_adjgamma_time, x_size,5, 'adjgamma')
showtimes(py_nonmax_time, pyCUDA_nonmax_time, x_size,6, 'nonmaxsuppression')

plt.show()





	
		
















