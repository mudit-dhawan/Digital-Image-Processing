## Importing required libraries
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import math


## Question 1 
noise_image = skimage.io.imread(fname="MuditDhawan_2018159_noiseIm.jpg")

plt.figure()
plt.imshow(noise_image, cmap='gray')
plt.title("Q1- Noisy image")
plt.show(block=False)


def create_box_filter(size):
    
    return np.ones(size)/(size[0]*size[1])

laplacian_filter_3x3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

h = create_box_filter((11, 11))


def pad(img, out_dim):
    
    out = np.zeros(out_dim)
    out[:img.shape[0], :img.shape[1]] = img
    
    return out

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


def cls_filter(g, h, l, _lambda, plot=True):
    
    ## Dimension of the matrix
    img_size = g.shape

    h = pad(h, img_size)
    l = pad(l, img_size)
    
    ## DFT 
    G = np.fft.fft2(g)
    H = np.fft.fft2(h)
    L = np.fft.fft2(l)
    
    if plot:
        plot_magnitude_spectrum(G, title="Magnitude spectrum of the Noisy Image")

        plot_magnitude_spectrum(H, title="Magnitude spectrum of the Box Filter")

        plot_magnitude_spectrum(L, title="Magnitude spectrum of the Laplacian")
    
    F_cap = np.conjugate(H)  / (np.absolute(H)**2 + _lambda*(np.absolute(L)**2)) 
    
    if plot: 
        plot_magnitude_spectrum(F_cap, title=f"Magnitude spectrum of the CLS Filter for lambda: {_lambda}")
    
    F_cap = F_cap * G
    ## Inverse DFT
    f_cap = np.real(np.fft.ifft2(F_cap))
    
    f_cap = (f_cap - f_cap.min()) / (f_cap.max() - f_cap.min())
    
    return f_cap * 255

clean_image = skimage.io.imread(fname="MuditDhawan_2018159_clean_image.jpg")


def mse(arr1, arr2):
    
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)
    
    mse = np.sum((arr1 - arr2)**2) / (arr1.shape[0] * arr1.shape[1])
    return mse


def psnr(ref_image, test_image):
    
    mse_value = mse(ref_image, test_image) 
    
    psnr_value = 10 * math.log10(255**2 / mse_value)
    
    return psnr_value


def tune_lambda(noise_image, h, laplacian_filter_3x3, ref_image, range_lambda=[0, 0.25, 0.5, 0.75, 1.0]):
    
    max_psnr = 0.0
    curr_best_lambda = -1
    
    for candidate_lambda in range_lambda:
        
        cls_clean_image = cls_filter(noise_image, h, laplacian_filter_3x3, candidate_lambda, plot=False)
        
        psnr_value = psnr(ref_image, cls_clean_image)

        if max_psnr < psnr_value:
            curr_best_lambda = candidate_lambda
            max_psnr = psnr_value
        
        print(f" Lambda Value: {candidate_lambda} | PSNR Value: {psnr_value}")
    cls_clean_image = cls_filter(noise_image, h, laplacian_filter_3x3, curr_best_lambda)
    
    plt.figure()
    plt.imshow(cls_clean_image, cmap='gray')
    plt.title(f"Best restored image fot \lambda = {curr_best_lambda}")
    plt.show(block=False)
    
    return curr_best_lambda

tune_lambda(noise_image, h, laplacian_filter_3x3, clean_image, range_lambda=[0, 0.25, 0.5, 0.75, 1.0])

input()

##########################################################################################################

## Question 3

image = skimage.io.imread(fname="MuditDhawan_2018159_inputImage.tif")
image = image.astype(float)
image = image / 255 ## Normalized image

## Plot input image
plt.figure()
plt.imshow(image)
plt.title("Question 3: Input Image")
plt.show(block=False)


def findSaturation(image):
    
    return (1 - ((3 * image.min(axis=-1)) / (image.sum(axis=-1) + 1e-3))) 

S = findSaturation(image)

def findIntensity(image):
    
    return (image.sum(axis=-1) / 3)


I = findIntensity(image)


def findHue(image):
    
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    
    theta = (((R - G) + (R - B)) / 2) / (((R - G)**2 + ((R - B) * (G - B)))**0.5 + 1e-3)
    
    theta = np.degrees(np.arccos(theta))

    mask = B <= G
    
    return (mask*theta + ((1 - mask) * (360 - theta)))

H = findHue(image)


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

def histogram_equalization(image, max_intensity=255, plot=True, uint_type=True):

    if uint_type:
        image = (image * 255).astype("uint8")
        
    ## Plot the input image 
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.grid()
    plt.title("Input Intensity Question 3")
    plt.show(block=False)

    h, intensity_values = find_normalised_histogram(image, plot=plot, 
                                                    title="Normalized Histogram for Intensity")
    
    H = np.cumsum(h)
    
    if plot:
        plt.figure()
        plt.plot(H)
        plt.grid()
        plt.title("CDF for Intensity for Question 3")
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
        plt.title("Histogram Equalized Intensity - Question 3")
        plt.grid()
        plt.show(block=False)
    
    h_dash, intensity_values_dash = find_normalised_histogram(output_image, plot=plot,
                                                            title="Normalized Histogram for Intensity - Histogram Equalized")
    
    return output_image

I_eq = histogram_equalization(I) / 255 

def HSItoRGB(H, S, I):

    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
    
    RG_mask = H < 120 
    B_RG = I * (1 - S) * RG_mask
    R_RG = I * (1 + ((S * np.cos(np.radians(H))) / np.cos(np.radians(60 - H)))) * RG_mask
    G_RG = (3*I - (R_RG + B_RG)) * RG_mask
    
    
    GB_mask = (H >= 120) * (H < 240) 
    H_GB = H - 120
    R_GB = I * (1 - S) * GB_mask
    G_GB = I * (1 + ((S * np.cos(np.radians(H_GB))) / np.cos(np.radians(60 - H_GB)))) * GB_mask
    B_GB = (3*I - (R_GB + G_GB)) * GB_mask

    
    BR_mask = H >= 240 
    H_BR = H - 240
    G_BR = I * (1 - S) * BR_mask
    B_BR = I * (1 + ((S * np.cos(np.radians(H_BR))) / np.cos(np.radians(60 - H_BR)))) * BR_mask
    R_BR = (3*I - (G_BR + B_BR)) * BR_mask

    
    R = R_RG + R_GB + R_BR
    G = G_RG + G_GB + G_BR
    B = B_RG + B_GB + B_BR
    
    R = R / np.max(R)
    G = G / np.max(G)
    B = B / np.max(B)
    
    image = np.concatenate((np.expand_dims(R, axis=2), np.expand_dims(G, axis=2), np.expand_dims(B, axis=2)), axis=-1)
    
    return image

new_image = HSItoRGB(H, S, I_eq)

plt.figure()
plt.imshow(new_image)
plt.title("HSI -> I Equalization -> RGB")
plt.show(block=False)

recon_image = HSItoRGB(H, S, I)

plt.figure()
plt.imshow(recon_image)
plt.title("HSI -> RGB (Reconstruction)")
plt.show(block=False)

input()