## Importing required libraries
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil, radians, cos, sin
from tqdm.auto import tqdm

## Code for Question 3 and 4

def _map_input_space(coord,T):
    """
    Function to map output-space co-ordinate to the input space by using inverse(T)
    
    Args:
        coord: Co-ordinate in the ouput space
        T: Transformation matrix
    
    Returns:
        (float, float)
    """
    
    ## Co-ordinate to map backwards
    X = np.transpose(np.array([coord[0], coord[1], 1]))
    
    ## Co-ordinate in input space
    V = np.dot(X, np.linalg.pinv(T))
    
    return (V[0], V[1])


def _check_integer(x):
    """
    Check if the input is an integer with the 1e-5 rounding 
    
    Args:
        x: input number 
    
    Return
        Boolean
    """
    x_int = round(x)

    if abs(x - x_int) < 1e-5:
        return True
    else:
        return False


def find_nn(coord, T, x_in_start, y_in_start):
    """
    Find the 4 nearest neighbours for a given point in output space and the given point mapped to input space 
    
    Args: 
        coord: Co-ordinate in the ouput space
        T: Transformation matrix
        x_in_start: x co-ordinate for the top left value of the input matrix
        y_in_start: y co-ordinate for the top left value of the input matrix
        
    Returns:
        scaled_coord: the given point mapped to input space
        x: numpy array containing x co-ordinates of the 4 nearest neighbours
        y: numpy array containing y co-ordinates of the 4 nearest neighbours
    """
    scaled_coord = _map_input_space(coord,T)
    scaled_coord = (scaled_coord[0] - x_in_start, scaled_coord[1] - y_in_start)

    x, y = [], []

    if _check_integer(scaled_coord[0]):
        if round(scaled_coord[0]) == 0:
            x1, x2 = 0, 1
        else:
            x1, x2 = round(scaled_coord[0])-1, round(scaled_coord[0])
    else:
        x1, x2 = floor(scaled_coord[0]), ceil(scaled_coord[0])
    
    if _check_integer(scaled_coord[1]):
        if round(scaled_coord[1]) == 0:
            y1, y2 = 0, 1
        else:
            y1, y2 = round(scaled_coord[1])-1, round(scaled_coord[1])
    else:
        y1, y2 = floor(scaled_coord[1]), ceil(scaled_coord[1])

    x = list(map(int, [x1, x1, x2, x2]))
    y = list(map(int, [y1, y2, y1, y2]))

    return scaled_coord, np.array(x), np.array(y) 


def check_within_image(x, y, width, height):
    """
    Checks if we have intensity values for the given co-ordinates.
    
    Args:
        x: numpy array containing x co-ordinates of the 4 nearest neighbours
        y: numpy array containing y co-ordinates of the 4 nearest neighbours
        width: width of the patch for which we have values 
        height: height of the patch for which we have values
    
    Returns:
        Boolean
    """
    
    if any(single_x >= width for single_x in x):
        return False
    
    if any(single_x < 0 for single_x in x):
        return False
    
    if any(single_y >= height for single_y in y):
        return False
    
    if any(single_y < 0 for single_y in y):
        return False

    return True


def _bilinear_interpolation(x, y, V):
    """
    Calculates parameters for the learnt bi-linear interpolation
    
    Args:
        x: numpy array containing x co-ordinates 
        y: numpy array containing y co-ordinates 
        V: intensity value after bi-linear interpolation
    
    Returns
        A: paramter matrix for bi-linear interpolation
    """
    
    X = np.concatenate((x, y, np.multiply(x,y), np.ones_like(x)), axis=1)

    A = np.dot(np.linalg.pinv(X), V)
    
    return A


def mirroring_edge(last_x, last_y, arr_out):
    """
    Function to perform mirroring to fill the border points in scaled image 
    
    Args:
        last_x: largest filled x co-ordinate of the scaled image
        last_y: largest filled y co-ordinate of the scaled image
        arr_out: matrix to be filled
        
    Returns:
        arr_out: mirror padded matrix
    """

    row_to_reflect = arr_out[last_x, :last_y]
    col_to_reflect = arr_out[:last_x, last_y]

    for reflect_row in range(last_x+1, arr_out.shape[0]):
        arr_out[reflect_row, :last_y] = row_to_reflect 

    for reflect_col in range(last_y+1, arr_out.shape[1]):
        arr_out[:last_x, reflect_col] = col_to_reflect 
    
    return arr_out


def _fill_nan_scaling(arr):
    """
    Fills nan values in a matrix on the borders with mirrored edge values 
    
    Args:
        arr: input matrix
    
    Returns:
        arr: filled output matrix
    """
    all_indices = np.where(~np.isnan(arr))
    last_x, last_y = np.max(all_indices[0]), np.max(all_indices[1])
    
    arr = mirroring_edge(last_x, last_y, arr)

    arr[np.where(np.isnan(arr))] = arr[last_x, last_y]

    return arr


def _fill_nan(arr, transformation_type='mixed'):
    """
    Fills nan values in a matrix with 0 (black) or mirrors edge values (scalling only)
    
    Args:
        arr: input matrix
    
    Returns:
        arr: filled output matrix
    """

    if transformation_type == 'scaling':
        arr = _fill_nan_scaling(arr)
    else:
        arr[np.where(np.isnan(arr))] = 0
    
    return arr


def _real_space_mapping(arr_out_size, x_start, y_start):
    """
    Finds the co-ordinate space for the ouput transformed matrix
    
    Args:
        arr_out_size: (height, width) of the output space
        x_start: x co-ordinate for the top left value of the output matrix
        y_start: y co-ordinate for the top left value of the output matrix
    
    Returns:
        arr_x: x co-ordinates of the output space
        arr_y: y co-ordinates of the output space
        coord_boundaries: co-ordinate boundaries for the output space (xmin, xmax, ymin, ymax)
    """
    arr_x = np.zeros(arr_out_size)
    arr_y = np.zeros(arr_out_size)
    
    coord_boundaries = [x_start, 0, 0, y_start]
    for row in range(arr_out_size[0]):
        arr_x[row, :] = y_start
        y_start += 1

    for col in range(arr_out_size[1]):
        arr_y[:, col] = x_start
        x_start += 1
    
    coord_boundaries[1] = x_start - 1
    coord_boundaries[2] = y_start - 1

    return arr_x, arr_y, coord_boundaries 


def bilinear_interpolation(arr, T, arr_out_size, x_start, y_start, x_in_start, y_in_start, transformation_type='mixed'):
    """
    Function to perform transformation and bi-linear interpolation for a given input matrix
    
    Args:
        arr: input numpy array
        T: Transformation matrix
        arr_out_size: output space dimension (height, width)
        x_start: x co-ordinate for the top left value of the output matrix
        y_start: y co-ordinate for the top left value of the output matrix
        x_in_start: x co-ordinate for the top left value of the input matrix
        y_in_start: y co-ordinate for the top left value of the input matrix
    
    Returns:
        arr_out: Output transformer matrix
        coord_boundaries: co-ordinate boundaries for the output space (xmin, xmax, ymin, ymax)
    """

    in_width = arr.shape[0]
    in_height = arr.shape[1]
    
    arr_out = np.ones(arr_out_size) *  np.nan
    arr_x, arr_y, coord_boundaries = _real_space_mapping(arr_out_size, x_start, y_start)

    for row in tqdm(range(arr_out.shape[0])):
        for col in range(arr_out.shape[1]):
            
            curr_coord = (arr_x[row, col], arr_y[row, col])

            scaled_curr_coord, x, y = find_nn(curr_coord, T, x_in_start, y_in_start)
            
            ## border element
            if not check_within_image(x, y, in_width, in_height):
                continue 

            x = np.minimum(x, np.ones_like(x)*(in_width-1))
            y = np.minimum(y, np.ones_like(x)*(in_height-1))

            V = arr[x, y]     

            A = _bilinear_interpolation(x.reshape((-1,1)), y.reshape((-1,1)), V.reshape((-1,1)))

            interpolated_ip = np.array([scaled_curr_coord[0], scaled_curr_coord[1], scaled_curr_coord[0]*scaled_curr_coord[1], 1])

            arr_out[row, col] = np.dot(interpolated_ip, A)
        
    arr_out = _fill_nan(arr_out, transformation_type)

    return arr_out, coord_boundaries


def scaling_output_size(in_width, in_height, T):
    """
    Function to create output size and co-ordinate map for scalling transformation
    
    Args:
        in_width: number of rows of input matrix
        in_height number of columns of input matrix
        T: Transformation matrix
    
    Return:
        arr_out_size: Ouput matrix size
        x_start: top left x co-ordinate for the output image
        y_start: top left y co-ordinate for the output image
    """

    factor_width, factor_height = T[0,0], T[1,1]

    arr_out_size = (ceil(in_width*factor_width), ceil(in_height*factor_height))
    
    x_start, y_start = 0, 0

    return arr_out_size, x_start, y_start


def translation_output_size(in_width, in_height, T):
    """
    Function to create output size and co-ordinate map for translation transformation
    
    Args:
        in_width: number of rows of input matrix
        in_height number of columns of input matrix
        T: Transformation matrix
        
    Return:
        arr_out_size: Ouput matrix size
        x_start: top left x co-ordinate for the output image
        y_start: top left y co-ordinate for the output image
    """
    
    factor_width, factor_height = T[2, 0], T[2,1]

    arr_out_size = (ceil(in_width+factor_width), ceil(in_height+factor_height))

    x_start = 0 if factor_width >=0 else factor_width
    y_start = 0 if factor_height >=0 else factor_height

    return arr_out_size, x_start, y_start


def rotation_output_size(in_width, in_height, T):
    """
    Function to create output size and co-ordinate map for rotation transformation
    
    Args:
        in_width: number of rows of input matrix
        in_height number of columns of input matrix
        T: Transformation matrix
        
    Return:
        arr_out_size: Ouput matrix size
        x_start: top left x co-ordinate for the output image
        y_start: top left y co-ordinate for the output image
    """
    
    diagonal = (in_width**2 + in_height**2)**0.5

    arr_out_size = (ceil(diagonal*2), ceil(diagonal*2))

    x_start, y_start = -ceil(diagonal), -ceil(diagonal)

    return arr_out_size, x_start, y_start


def mixed_output_size(in_width, in_height, T):
    """
    Function to create output size and co-ordinate map for mixed transformation
    
    Args:
        in_width: number of rows of input matrix
        in_height number of columns of input matrix
        T: Transformation matrix
        
    Return:
        arr_out_size: Ouput matrix size
        x_start: top left x co-ordinate for the output image
        y_start: top left y co-ordinate for the output image
    """

    scaling_factor = ceil(max(abs(T[0,0]), abs(T[1,1]), abs(T[0,1]), abs(T[1,0])))

    factor_width, factor_height = T[2, 0], T[2,1]
    x_start_translation = 0 if factor_width >=0 else factor_width
    y_start_translation = 0 if factor_height >=0 else factor_height

    diagonal = ((in_width*scaling_factor)**2 + (in_height*scaling_factor)**2)**0.5

    arr_out_size = (ceil(factor_width+diagonal*2), ceil(factor_height+diagonal*2))

    x_start_rotation, y_start_rotation = -ceil(diagonal), -ceil(diagonal)

    x_start = x_start_translation + x_start_rotation
    y_start = y_start_translation + y_start_rotation

    return arr_out_size, x_start, y_start


def transformation(arr, T, transformation_type='mixed'):
    """
    Function to perform Transformation on the input matrix based on the input trasformation matrix 
    
    Args:
        arr: input matrix
        T: Transformation matrix
        transformation_type: if a specific type of transformation is used (scaling -> mirroring edge points)
    Returns:
        arr_out: Output transformer matrix
        coord_boundaries: co-ordinate boundaries for the output space (xmin, xmax, ymin, ymax)
    """

    in_width, in_height = arr.shape[0], arr.shape[1]

    if transformation_type == 'scaling':
        arr_out_size, x_start, y_start = scaling_output_size(in_width, in_height, T)
        
    elif transformation_type == 'translation':
        arr_out_size, x_start, y_start = translation_output_size(in_width, in_height, T)
        
    elif transformation_type == 'rotation':
        arr_out_size, x_start, y_start = rotation_output_size(in_width, in_height, T)
        
    elif transformation_type == 'mixed':
        arr_out_size, x_start, y_start = mixed_output_size(in_width, in_height, T)
        
    else:
        print(f"We don't support {key} transformation.")
        return 
            
    arr_out, coord_boundaries = bilinear_interpolation(arr, T, arr_out_size, x_start, y_start, 0, 0, transformation_type)

    return arr_out, coord_boundaries


def scaling(scale_x, scale_y):
    """
    Function create a transformation matrix based on scaling factor for x and y axes.
    
    Args:
        scale_x: scaling factor for x axes
        scale_y: scaling factor for y axes
        
    Returns:
        T: scaling transformation matrix
    """
    T = np.array([[scale_x, 0, 0], 
                  [0, scale_y, 0], 
                  [0, 0, 1]])

    return T


def translation(shift_x, shift_y):
    """
    Function create a transformation matrix based on shifting factor for x and y axes.
    
    Args:
        shift_x: shifting factor for x axes
        shift_y: shifting factor for y axes
        
    Returns:
        T: translation transformation matrix
    """
    T = np.array([[1, 0, 0], 
                  [0, 1, 0], 
                  [shift_x, shift_y, 1]])
    
    return T


def rotation(degree):
    """
    Function to create a transformation matrix based on degree pf rotation
    
    Args:
        degree: degree of rotation
        
    Returns:
        T: rotation transformation matrix
    """
    
    rad = radians(degree)
    
    T = np.array([[cos(rad), -sin(rad), 0], 
                  [sin(rad), cos(rad), 0], 
                  [0, 0, 1]])
    
    return T


def create_transformation_matrix(transformation_list):
    """
    Function to create transformation matrix for a given list of transformations
    
    Args:
        transformation_list: list of transformations to be performs in order
    
    Returns:
        T: transformation matrix
    """
    T = np.identity(3)
    
    for item in transformation_list:
        key, val = item
        
        if key == 'scaling':
            T = np.matmul(T, scaling(*val))
        elif key == 'translation':
            T = np.matmul(T, translation(*val))
        elif key == 'rotation':
            T = np.matmul(T, rotation(val))
        else:
            print(f"We don't support {key} transformation.")
        
    return T

## Code to drive Question 3
print("---------------------------------------------------------------")
print("Output for Question 3:")

## Input arrays for testing question 3
# a = np.array([[1,2,3], [4,5,6], [7,8,9]])
a = np.array([[1, 3, 5, 7], [7, 9, 11, 13], [13, 15, 17, 19]])
# a = np.array([[180,2], [3,4]])
print("Input Matrix for Question 2 Matrix: \n", a) 

## Scaling factor for testing question 3
scaling_factor = 1.5
print("The interpolation factor for Question 3:", scaling_factor)
transformation_list = [("scaling", (scaling_factor, scaling_factor))]
T_scaling = create_transformation_matrix(transformation_list)

print("Transformation Matrix: \n", T_scaling)
## Could be called without the 3rd argument ("scaling"), but it would 
## consider a larger output space and make them as black pixels. 
transformed_matrix, matrix_coord_space = transformation(a, T_scaling, "scaling")
print("Transformed Matrix: \n", transformed_matrix) 
print("---------------------------------------------------------------")


## Code to drive Question 4
print("---------------------------------------------------------------")

## Read the Image 
image = skimage.io.imread(fname="MuditDhawan_2018159_InputImage.jpg")

## Plot the input image 
print("Input Image/ Matrix for Question 4:")
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Input Image for Question 4 and Reference image for Question 5')
plt.show(block=False)

## To block and view the image
input()

## Create the transformation matrix
transformation_list = [("rotation", 45), ("scaling", (2,2)), ("translation", (30, 30))]
T = create_transformation_matrix(transformation_list)
print("Transformation Matrix for Question 4: \n", T)

## Find the transformed matrix -- output matrix, output space co-ordinates 
transformed_image, coord_boundaries = transformation(image,T)

## Plot the transformed image
print("Transformed Image:")
plt.figure()
plt.imshow(transformed_image/255, cmap='gray', extent=coord_boundaries)
plt.title('Output Transformed Image for Question 4 and Input for Question 5')
plt.grid()
plt.show(block=False)
print("---------------------------------------------------------------")

## To block and view the image
input()

## Extra functions for Question 5 - Image registration

def find_transformation(x, y, v, w):
    """
    Function to find transfromation matrix based on the output and input image's common points
    
    Args:
        x: numpy array containing x co-ordinates (Output Image)
        y: numpy array containing y co-ordinates (Output Image)
        v: numpy array containing x co-ordinates (Input Image)
        w: numpy array containing y co-ordinates (Input Image)
    
    Retruns:
        T: Tranformation matrix for V -> (T) -> X
    """
    X = np.concatenate((x, y, np.ones_like(x)), axis=1)
    
    V = np.concatenate((v, w, np.ones_like(v)), axis=1)
    
    T = np.dot(np.linalg.pinv(V), X)
    
    return T


def image_registration(O, I, landmark_pts, x_in_start, y_in_start):
    """ 
    Function to register the output image wrt to a reference image based on landmark (common) points
    
    Args:
        O: the unregistered input image
        ref: the reference image
        landmark_pts: landmark points (x, y, v, w)
        x_in_start: top left x co-ordinate for the input image
        y_in_start: top left y co-ordinate for the input image
    
    Returns:
        registered_ip_image: registered input image
    """
    x, y, v, w = landmark_pts
    
    Z = find_transformation(x, y, v, w)
    
    print("Computed Transformation Matrix for Question 5: \n", T)
    ref_size = I.shape
    
    registered_ip_image = bilinear_interpolation(O, np.linalg.pinv(Z), ref_size, 
                                                 0, 0, x_in_start, y_in_start, 
                                                 transformation_type='mixed')
    
    return registered_ip_image


# Common points in the two images 
# 
# (-410, 567), (91, 315), (-383, 913), (413, 990) -> output image (X)
# 
# (34, 345), (123, 79), (166, 458), (476, 203) -> reference image (V)

## Code to drive Question 5
print("---------------------------------------------------------------")

## forming the landmark points 
x = np.array([567, 315, 913, 990]).reshape((-1, 1)) ## y co-ordinates of the common points output image
y = np.array([-410, 90, -383, 413]).reshape((-1, 1)) ## x co-ordinates of the common points output image

v = np.array([345, 79, 458, 203]).reshape((-1, 1)) ## y co-ordinates of the common points input image
w = np.array([34, 123, 166, 476]).reshape((-1, 1)) ## y co-ordinates of the common points input image

landmark_pts = x, y, v, w

## Computing the registered image
registered, registered_coord_boundaries = image_registration(transformed_image, image, landmark_pts, 
                                                             coord_boundaries[0], coord_boundaries[-1])


## Plot the registered image image
plt.figure()
plt.imshow(registered, cmap='gray', extent=registered_coord_boundaries)
plt.title('Register Image: Output for Question 5')
plt.grid()
plt.show(block=False)
plt.show()
print("---------------------------------------------------------------")