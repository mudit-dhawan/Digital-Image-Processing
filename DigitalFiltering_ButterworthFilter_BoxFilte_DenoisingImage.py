## Importing required libraries
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy.signal

#####################################################################
## Code for Question 1
#####################################################################

def plot_magnitude_spectrum(dft, title="Magnitude spectrum of the Image"):
    
    ## Calculate magnitude spectrum 
    dft_mag = np.absolute(dft)
    
    ## convert to log scale
    log_mag = np.log(dft_mag + 1)
    
    ## normalize the output
    normalized_log_mag = log_mag / np.amax(log_mag)
    
    ## Plot the magnitude spectrum
    plt.figure()
    plt.imshow(normalized_log_mag*255, cmap='gray')
    plt.title(title)
    # plt.grid()
    plt.show(block=False)

def create_BW_filter(size, threshold, order=2):
    
    ## Find center of the array 
    n, m = size[0]/2, size[1]/2 
    
    ## create a dummy array with x, y coordinates
    arr_x = np.zeros(size)
    arr_y = np.zeros(size)
    
    ## top left corner 
    x_start, y_start = 0., 0.
    
    for row in range(size[0]):
        arr_x[row, :] = y_start - m ## subtract the cetner coordinates to perform centering
        y_start += 1

    for col in range(size[1]):
        arr_y[:, col] = x_start - n ## subtract the cetner coordinates to perform centering
        x_start += 1
    
    ## D(u,v)
    d = (arr_x**2 + arr_y**2)**0.5
    
    ## impulse respone of the Butterwoth filter
    H = 1 / (1+ (d/threshold)**(2*order))
    
    return H


def centered_shift(image):
    
    centered_image = image.copy()
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            centered_image[row, col] = image[row, col]*((-1)**(row+col))
    
    return centered_image


def dft(image, shift=True):
    
    if shift:
        image = centered_shift(image)
    
    image_dft = np.fft.fft2(image)
    
    return image_dft


def display_bw_filter(bw_filter_response, threshold, order=2):
    
    bw_filter = np.real(np.fft.ifft2(bw_filter_response))
    
    bw_filter = centered_shift(bw_filter)
    bw_filter = bw_filter[:int(bw_filter.shape[0]/2), :int(bw_filter.shape[1]//2)]
    
    ## plot the Butterworth filter
    plt.figure()
    plt.imshow(bw_filter, cmap='gray')
    plt.title(f"Butterworth filter threshold: {threshold}, order: {order}")
    plt.show(block=False)
    
    ## plot the Butterworth filter
    plt.figure()
    plt.imshow(bw_filter[:10, :10], cmap='gray')
    plt.title(f"Left most 10x10 Butterworth filter threshold: {threshold}, order: {order}")
    plt.show(block=False)
    
    ## Plot centered magnitude response of the filter
    plot_magnitude_spectrum(bw_filter_response, 
                            title=f"Centered Magnitude response of the BW Filter | threshold: {threshold}, order: {order}")


def q1(image_name="MuditDhawan_2018159_input image.jpg", thresholds=[10, 30, 60]):
    
    image = skimage.io.imread(fname=image_name)
    image = image.astype(float)
    
    ## Plot input image
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("Question 1: Input Image")
    plt.show(block=False)
    
    ## Pad the image: (n,m) -> (2n,2m) 
    image_padded = np.zeros((image.shape[0]*2, image.shape[1]*2))
    
    image_padded[:image.shape[0], :image.shape[1]] = image
    
    ## Plot Padded input image
    plt.figure()
    plt.imshow(image_padded, cmap='gray')
    plt.title("Padded Image  for Q1")
    plt.show(block=False)
    
    ## dft of the Input Padded Image
    image_dft_ = dft(image_padded, shift=False)
    plot_magnitude_spectrum(image_dft_, title="Magnitude response of the Input Padded Image")
    
    ## Centered DFT of the Input Padded Image
    image_dft = dft(image_padded, shift=True)
    plot_magnitude_spectrum(image_dft, title="Centered Magnitude response of the Input Padded Image")
    
    for threshold in thresholds:
        ## Create imupulse response for the BW filter 
        bw_filter_response = create_BW_filter(image_padded.shape, threshold, order=2)
        
        ## Display the filter
        display_bw_filter(bw_filter_response, threshold, order=2)
        
        ## Element-wise multiplication
        output_ = image_dft * bw_filter_response
        
        plot_magnitude_spectrum(output_, 
                                title=f"Centered Magnitude response of the Output Image| threshold: {threshold}, order: 2")
        
        ## Inverse dft
        output = np.fft.ifft2(output_)
        
        ## Inverse Centering the output
        output = centered_shift(output)
        
        ## Plot input image
        plt.figure()
        plt.imshow(np.real(output)[:image.shape[0], :image.shape[1]], cmap='gray')
        plt.title(f"Question 1: Output Image | threshold: {threshold}, order: 2")
        plt.show(block=False)


## Calling the function for Question 1
q1()
input() ## Block the code

###################################################################################################################


#####################################################################
## Code for Question 2
#####################################################################

def plot_image(image, title="Image"):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show(block=False)


def pad(img, out_dim):
    
    out = np.zeros(out_dim)
    out[:img.shape[0], :img.shape[1]] = img
    
    return out


def conv_freq(image, kernel):
    
    ## Dimension of the two matrices
    n1, m1 = image.shape
    n2, m2 = kernel.shape 
    
    ## Convolution output size
    conv_out_size = (n1 + n2 - 1, m1 + m2 - 1)
    
    ## Pad the image and kernel
    image_padded = pad(image, conv_out_size)
    kernel_padded = pad(kernel, conv_out_size)
    
    ## DFT 
    image_dft = dft(image_padded, shift=False)
    kernel_dft = dft(kernel_padded, shift=False)
    
    ## Element wise multiplication
    out = image_dft * kernel_dft
    
    ## Inverse DFT
    out = np.fft.ifft2(out)
    
    ## Return real part 
    return np.real(out)


def create_box_filter(size):
    
    return np.ones(size)/(size[0]*size[1])


def q3(image_name="MuditDhawan_2018159_input image.jpg"):
    image = skimage.io.imread(fname=image_name)
    image = image.astype(float)
    
    ## Plot input image
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("Question 3: Input Image")
    plt.show(block=False)
    
    ## Convolution using my own code
    conv_output_own = conv_freq(image, create_box_filter((9,9)))
    
    print("9X9 Box filter: ")
    print(create_box_filter((9,9)))
    
    ## Plot Conv Output
    plt.figure()
    plt.imshow(conv_output_own, cmap='gray')
    plt.title("Question 3: Output Image using my own code for convolution")
    plt.show(block=False)
    
    out = scipy.signal.convolve2d(image, create_box_filter((9,9)))
    
    ## Plot Conv Output
    plt.figure()
    plt.imshow(out, cmap='gray')
    plt.title("Question 3: Output Image using Scipy for convolution")
    plt.show(block=False)

## Calling the function for Question3 
q3()
input() ## Block the code

###################################################################################################################


#####################################################################
## Code for Question 4
#####################################################################

def q4():
    
    ## Read the Noisy Image 
    image_n = skimage.io.imread(fname="MuditDhawan_2018159_noiseIm.jpg")
    image_n = image_n.astype(float)
    
    ## Plot the image 
    plt.figure()
    plt.imshow(image_n, cmap='gray')
    plt.title("Question4: Noisy Image")
    plt.show(block=False)
    
    ## DFT 
    image_dft_n = dft(image_n, shift=True)
    
    plot_magnitude_spectrum(image_dft_n, title="Centered Image spectrum of the noisy image") 
    
    ## Read the De-Noised Image 
    image_d = skimage.io.imread(fname="MuditDhawan_2018159_denoiseIm.jpg")
    image_d = image_d.astype(float)
    
    ## Plot the image 
    plt.figure()
    plt.imshow(image_d, cmap='gray')
    plt.title("Question4: Denoised Image")
    plt.show(block=False)
    
    ## DFT 
    image_dft_d = dft(image_d, shift=True)
    
    plot_magnitude_spectrum(image_dft_d, title="Centered Image spectrum of the Denoised image") 
    
    ## There is a white spot at (160, 160) and (96, 96) as the this is a centered magnitude
    ## spectrum plot, therefore the sinusoidal noise added has the frequency of 32 units
    
    ## Removing the fruency around the added noise frequency (+- 5 units) 
    filter_denoise = np.ones((image_dft_n.shape[0], image_dft_n.shape[1]))
    filter_denoise[91:101, 91:101] = 0
    filter_denoise[155:165, 155:165] = 0
    
    ## De-noising the image
    denoised_image_dft = image_dft_n * filter_denoise
    
    plot_magnitude_spectrum(denoised_image_dft, title="Centered Image spectrum of the Noisy image after applying the filter")
    
    out = np.fft.ifft2(denoised_image_dft)
    out = centered_shift(out)
    
    ## Plot the denoised image after applying the filter
    plt.figure()
    plt.imshow(np.real(out), cmap='gray')
    plt.title("Question4: Ouput Denoised Image")
    plt.show(block=False)

## Calling function for Question 4
q4()
input() ## Block the code

###################################################################################################################