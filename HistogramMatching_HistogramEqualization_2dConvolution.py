## Importing required libraries
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil, radians, cos, sin
from tqdm.auto import tqdm

###############################################################################################

## Functions for Question 3
def find_normalised_histogram(image, plot=True, title="Normalized Histogram for the image"):
    intensity_values, intensity_freq = np.unique(image, return_counts=True)
    
    h = intensity_freq / (image.shape[0]*image.shape[1])
    
    ## Plot the image
    if plot:
        plt.figure()
        plt.bar(x=intensity_values, height=h)
        plt.grid()
        plt.title(title)
        plt.show(block=False)
    
    return h, intensity_values

def histogram_equalization(image, max_intensity=255, plot=True):
    
    ## Plot the input image 
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.grid()
    plt.title("Input Image/ Matrix for Question 3")
    plt.show(block=False)

    h, intensity_values = find_normalised_histogram(image, plot=plot, 
                                                    title="Normalized Histogram for the Input image")
    
    H = np.cumsum(h)
    
    if plot:
        plt.figure()
        plt.plot(H)
        plt.grid()
        plt.title("CDF for the Input Image/ Matrix for Question 3")
        plt.show(block=False)
    
    s = max_intensity*H
    
    output_image = np.zeros_like(image)
    for idx, intensity_value in enumerate(intensity_values):
        indices = np.where(image == intensity_value)

        output_image[indices] = s[idx]
    
    ## Plot the image 
    if plot:
        plt.figure()
        plt.imshow(output_image, cmap='gray')
        plt.title("Histogram Equalized image - Ouput for Question 3")
        plt.grid()
        plt.show(block=False)
    
    h_dash, intensity_values_dash = find_normalised_histogram(output_image, plot=plot,
                                                            title="Normalized Histogram for the Output image - Histogram Equalized")
    
    return output_image

###############################################################################################

## Driver code Question 3

print("#######################################################################################")
print("Starting Question 3:")

## Read the Image 
image = skimage.io.imread(fname="MuditDhawan_2018159_InputImage.jpg")

## Call histogram equalization
output_image = histogram_equalization(image, max_intensity=255, plot=True)

input() ## Block the code
###############################################################################################

## Functions for Question 4
def gamma_transformation(I, gamma=0.5, plot=True):
    I = image.astype(np.float64)
    
    c = 255**(1-gamma)
    
    gamma_image = np.around(c*np.power(I, gamma))
    
    if plot:
        ## Plot the gamma image 
        plt.figure()
        plt.imshow(gamma_image, cmap='gray')
        plt.title(f"Gamma Transformed image for gamma: {gamma}")
        plt.grid()
        plt.show(block=False)
        
    return gamma_image

def histogram_matching(I, T, plot=True):
    
    ## Find normalized histogram for input and target images
    h, intensity_values_h = find_normalised_histogram(I, plot=plot, 
                                                    title="Normalized histogram for the input image")
    g, intensity_values_g = find_normalised_histogram(T, plot=plot, 
                                                    title="Normalized histogram for the Target image")
    
    ## Calculate CDF for both the images 
    H = np.cumsum(h)
    G = np.cumsum(g)
    
    ## Find the closest intensity values 
    closest = np.argmin(np.abs(H.reshape(-1, 1) - np.transpose(G)), axis=1)
    
    matched_image = np.zeros_like(I)
    for idx, closest_intensity_value_index in enumerate(closest):

        ## for all the pixels where intensity value is r
        indices = np.where(I == intensity_values_h[idx]) 

        ## replacing those pixels with s for which F(r) closest to G
        matched_image[indices] = intensity_values_g[closest_intensity_value_index] 
    
    ## Plot the image
    if plot:
        plt.figure()
        plt.imshow(matched_image, cmap='gray')
        plt.title("Histogram Matched Image")
        plt.grid()
        plt.show(block=False)
        
    return matched_image

def ques4(I, gamma=0.5, plot=True):
    
    ## Plot the input image 
    plt.figure()
    plt.imshow(I, cmap='gray')
    plt.grid()
    plt.title("Input Image/ Matrix for Question 4")
    plt.show(block=False)
    
    ## Perform Gamma Transformation
    T = gamma_transformation(I, gamma=gamma, plot=plot)
    
    ## Perform Histogram matching
    matched_image = histogram_matching(I, T)
    
    ## Find normalized histogram for histogram matched image
    g_dash, intensity_values_dash = find_normalised_histogram(matched_image, plot=plot, 
                                                    title="Normalized histogram for the Histogram Matched Output")
    
    return matched_image

###############################################################################################

## Driver code Question 4

print("#######################################################################################")
print("Starting Question 4:")

## Read the Image 
image = skimage.io.imread(fname="MuditDhawan_2018159_InputImage.jpg")

## Perform Histogram matching
matched_image = ques4(image)

input() ## Block the code
###############################################################################################

## Functions for Question 5

def convolution(image, kernel):
    
    img_x, img_y = image.shape
    
    rot_kernel = np.rot90(kernel, k=2)
    kernel_x, kernel_y = rot_kernel.shape
    
    out_x = image.shape[0] + kernel_x - 1
    out_y = image.shape[1] + kernel_y - 1
    
    pad_amt_x = kernel_x - 1 
    pad_amt_y = kernel_y - 1
      
    pad_image = np.zeros((img_x+(2*pad_amt_x), img_y+(2*pad_amt_y)))
    
    pad_image[pad_amt_x:pad_amt_x+img_x, pad_amt_y:pad_amt_y+img_y] = image
    
    output = np.zeros((out_x, out_y))
    
    for row in range(0,pad_image.shape[0]):
        for col in range(0,pad_image.shape[1]):
            
            if row+kernel_x > pad_image.shape[0]: ## out of width
                break
            
            elif col+kernel_y > pad_image.shape[1]: ## out of height
                break
            
            else:
                output[row, col] = (rot_kernel * pad_image[row:row+kernel_x, col:col+kernel_y]).sum()

    return output

###############################################################################################

## Driver code Question 5

print("#######################################################################################")
print("Starting Question 5:")


print("Checking convolution for the given example:")
I = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
filter_ =  np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Example Input Matrix:\n", I)
print("Example Filter Matrix:\n", filter_)

convolution_output = convolution(I, filter_)

print("Convolution Output for the example:\n ", convolution_output)

print("----------------------------------------------------------")

print("Question 5 for random matrices")

I = np.random.randint(low=0, high=10, size=(3,3))
filter_ = np.random.randint(low=0, high=10, size=(3,3))

print("Random Input Matrix:\n", I)
print("Random Filter Matrix:\n", filter_)

convolution_output = convolution(I, filter_)

print("Convolution Output for the Random input: \n", convolution_output)

##############################################################################################