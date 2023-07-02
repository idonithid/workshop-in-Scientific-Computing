'Ido Nitzan Hidekel, 322996364'
######################################## HW number 1 ##############################################
import zlib
import os
import time
import argparse
import random
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pyfftw
from PIL import Image
from skimage.metrics import structural_similarity as ssim
#########################################################
#   help function to generate complex numbers array!    #
#########################################################

def complex_array(arr_len):
    '''
    generates random complex array for testing

    Args:
    lenght of array

    Returns:
    complex np array of size arg
    '''
    lst = []
    cnt = 1
    while cnt <= arr_len:
        low = random.uniform(-1000, 1000)
        high = random.uniform(-1000, 1000)
        lst.append(complex(low, high))
        cnt +=1
    return np.array(lst)

def verify_input(input_list):
    """
    Verifies that the input list contains only complex or real numbers.

    Args:
    input_list (numpy array or list): The list to be verified.

    """
    # check that the vector is iterable
    try:
        iter(input_list)
    except TypeError as ex:
        raise ValueError('Input vector is not iterable') from ex

    for item in input_list:
        # check for wrong dtypes
        if not isinstance(item, (complex, float, int)):
            raise ValueError('Input vector contains variables that are not int,float or complex')


###############
#   task 1    #
###############
def dft_1d(vec, direction):
    '''
    Calculates 1 dim discrete Fourier transform, or its inverse on @vec
    '''
    verify_input(vec)
    arr_len = len(vec)
    result = np.array([0 for i in range(arr_len)], dtype=complex)
    if direction == 'FFTW_FORWARD':
        for k in range(arr_len):
            res = 0
            for j in range(arr_len):
                res += vec[j] * np.exp(complex(0, 2 * np.pi * k * j / arr_len))
            result[k] = res
        return result
    if direction == 'FFTW_BACKWARD':
        for k in range(arr_len):
            res = 0
            for j in range(arr_len):
                res += vec[j] * np.exp(complex(0, -2 * np.pi * k * j / arr_len))
            result[k] = res
        return 1 / arr_len * result
    return "Wrong direction"


###############
#   task 2    #
###############

def test_dft_1d(eps=2 ** -16):
    """"" 
    perform 3 tests to check my functions:
    """""

    vec = random.randint(20, 100)
    vec = complex_array(vec)


    #TEST 1
    #verfiy that preform DFT backwards on a DFT forward output return the original input


    expected_vec = dft_1d(dft_1d(vec, 'FFTW_FORWARD'), 'FFTW_BACKWARD')
    norm = np.linalg.norm(expected_vec - vec)
    if norm < eps:
        print('Inverse test PASS')
    else:
        print('Inverse test FAIL')
    #TEST 2
    #compare analytic solution with array
    input_test = [0, 1, 2, 3]
    ## my manual calculation of the dft for x
    analytic_dft = np.array([6, complex(-2, -2), -2, complex(-2, 2)], dtype=complex)

    ## my code for dft for x
    our_dft = dft_1d(input_test, "FFTW_FORWARD")
    error = np.linalg.norm(our_dft - analytic_dft)
    if error < eps:
        print("Test 2 PASS")

    #TEST 3
    #compare analytic solution with array
    our_dft_inverse = dft_1d(analytic_dft, "FFTW_BACKWARD")
    error = np.linalg.norm(our_dft_inverse - input_test)
    if error < eps:
        print("Test 3 PASS")


###############
#   task 3    #
###############

def dft_1d_matrix(vec, direction):
    " calculate dft with matrix multiply"
    verify_input(vec)

    if direction == 'FFTW_FORWARD':
        vec_len = len(vec)
        n_array = np.arange(vec_len)
        k_array = n_array.reshape((vec_len, 1))
        ## create the matrix
        fft_mat = np.exp(2j * np.pi * k_array * n_array / vec_len)

        # multiply the matrix with the input vector
        res = np.matmul(fft_mat, vec)
        return res
    # Backward DFT
    if direction == 'FFTW_BACKWARD':
        vec_len = len(vec)
        n_array = np.arange(vec_len)
        k_array = n_array.reshape((vec_len, 1))
        ## create the matrix
        fft_mat = np.exp(-2j * np.pi * k_array * n_array / vec_len)
        # multiply the matrix with the input vector
        res = np.matmul(fft_mat, vec)
        return 1 / vec_len * res
    return "Wrong direction"

###############
#   task 4    #
###############
def plot_dft_timing():
    'plotting'
    n_list = list(range(1,1000,100))
    dft_1_time_lst = np.zeros(len(n_list))
    dft_mat_time_lst = np.zeros(len(n_list))
    for i, vec_len in enumerate(n_list):
        vec = complex_array(vec_len)
        start_time = time.time()
        dft_1d(vec, 'FFTW_FORWARD')
        dft_1_time_lst[i] = time.time() - start_time
        start_time = time.time()
        dft_1d_matrix(vec, 'FFTW_FORWARD')
        dft_mat_time_lst[i] = time.time() - start_time

    # Draw the graph
    plt.plot(n_list, dft_1_time_lst, 'b-', label='dft_1d')
    plt.plot(n_list, dft_mat_time_lst, 'r-', label='dft_1d_matrix')

    # Calculate curve fit
    const1, alpha1 = curve_fit(estimated_func, n_list, dft_1_time_lst)[0]
    const2, alpha2 = curve_fit(estimated_func, n_list, dft_mat_time_lst)[0]

    # Draw the curve fit
    plt.plot(n_list, estimated_func(n_list, const1, alpha1), 'c-',
             label= f'dft_1d fit: c={const1}, a={alpha1}')
    plt.plot(n_list, estimated_func(n_list, const2, alpha2), 'k-',
             label= f'dft_1d_matrix fit: c={const2}, a={alpha2}')

    # Labeling the axis
    plt.xlabel('n')
    plt.ylabel('time')

    # Printing legend and constants and plot
    plt.legend()
    print("The constants of the manual calculation are:")
    print("c:", const1)
    print("a:", alpha1)

    print("The constants of the matrix calculation are:")
    print("c:", const2)
    print("a:", alpha2)

    plt.show()


def estimated_func(input_list, const, alpha):
    'estimate function'
    return const * (input_list ** alpha)

######################################## HW number 2 ##############################################

######################
#  helper functions  #
######################

def verify_input_2(input_list):
    """
    Verifies that the input list contains only complex or real numbers.

    Args:
    input_list (numpy array or list): The list to be verified.

    """
    # check that the vector is iterable
    try:
        iter(input_list)
    except TypeError as ex:
        raise ValueError('Input vector is not iterable') from ex
    input_len = len(input_list)
    if math.ceil(np.log2(input_len)) != math.floor(np.log2(input_len)):
        raise ValueError("The vector lenght is not a power of 2!")

    for item in input_list:
        # check for wrong dtypes
        if not isinstance(item, (complex, float, int)):
            raise ValueError('Input vector contains variables that are not int,float or complex')


def merge_sub_array(arr_1,arr_2,direction):
    '''
    hepler function for the divied and contour solution
    perform the dft calculation each step
    '''

    assert len(arr_1) == len(arr_2)
    full_len = 2 * len(arr_2)
    if direction == 'FFTW_FORWARD':
        const = np.exp(2 * np.pi * 1j / full_len)
        factor = 1
    elif direction == 'FFTW_BACKWARD':
        const = np.exp(-2 * np.pi * 1j / full_len)
        factor = 2 / full_len
    res = np.zeros(full_len,dtype=complex)

    # implement by the formula
    for k in range(full_len // 2):
        res[k] = arr_1[k] + arr_2[k] * (const ** k)
        res[k + full_len // 2] = arr_1[k] - arr_2[k] * (const ** k)
    return res*factor


######################
#       TASK 1       #
######################

def fft_1d_recursive_radix2(vec,direction):
    '''''
    compute the fft with recursive
    '''''
    # check the input vector
    verify_input_2(vec)
    vec_len = len(vec)
    if direction == 'FFTW_FORWARD':
        const = np.exp(1j*np.pi*2/vec_len)
    elif direction == 'FFTW_BACKWARD':
        const = np.exp(-1j * np.pi * 2 / vec_len)
    if direction not in ('FFTW_FORWARD', 'FFTW_BACKWARD'):
        raise ValueError('Invalid direction input')
    res = np.zeros(vec_len,dtype=complex)

    # the recursion base
    if vec_len==1:
        return [vec[0]]

    # divied the vector to odd and even
    vec_even = vec[::2]
    vec_odd = vec[1::2]

    # perform the recursive step
    fft_even = fft_1d_recursive_radix2(vec_even,direction)
    fft_odd =  fft_1d_recursive_radix2(vec_odd,direction)

    # implement by the formula
    for k in range(vec_len//2):
        res[k] = fft_even[k] + fft_odd[k]*(const**k)
        res[k+vec_len//2] = fft_even[k] - fft_odd[k]*(const**k)
    if direction == 'FFTW_BACKWARD':
        return (2/vec_len)*res
    return res

######################
#       TASK 2       #
######################
def fft_1d_radix2(vec,direction):
    '''''
        compute the fft using bit reverse
    '''''

    # check the input vector
    verify_input_2(vec)
    vec_len = len(vec)

    bit_rev = np.zeros(vec_len,dtype=complex)
    log_n = int(np.log2(vec_len))

    # reverse bits
    for i, elem in enumerate(vec):
        bit_str = bin(i)[2:]
        bit_str = bit_str.zfill(log_n)
        bit_str = bit_str[::-1]
        bit_str = int(bit_str, 2)
        bit_rev[bit_str] = elem

    res = bit_rev.copy()
    for power in range(1,log_n+1):
        res_copy = res.copy()
        window_size = 2**power

        for start_pt in range(0,vec_len,2**power):
            bit_rev_first_half = res[start_pt:start_pt+((window_size)//2)]
            bit_rev_second_half = res[start_pt +((window_size)//2):start_pt + window_size]
            res_copy[start_pt:start_pt+window_size] = \
                merge_sub_array(bit_rev_first_half,bit_rev_second_half,direction)
        res = res_copy.copy()
    return res

###############
#   task 3    #
###############
def plot_fft_radix2_timing():
    ''' plot time graph'''

    n_list = [2**i for i in range(1,15,5)]
    fft_rec_time_lst = np.zeros(len(n_list))
    fft_non_rec_time_lst = np.zeros(len(n_list))
    for i,vec_len in enumerate(n_list):
        vec = complex_array(vec_len)
        start = time.time()
        fft_1d_recursive_radix2(vec,'FFTW_FORWARD')
        fft_rec_time_lst[i]=time.time()-start

        start = time.time()
        fft_1d_radix2(vec, 'FFTW_FORWARD')
        fft_non_rec_time_lst[i]=time.time()-start

    # Draw the graph
    plt.plot(n_list, fft_rec_time_lst, 'b-', label='recursion')
    plt.plot(n_list, fft_non_rec_time_lst, 'r-', label='non-recursion')

    # Calculate curve fit
    const1 = curve_fit(estimated_func_2, n_list, fft_rec_time_lst)[0]
    const2 = curve_fit(estimated_func_2, n_list, fft_non_rec_time_lst)[0]

    # Draw the curve fit
    plt.plot(n_list, estimated_func_2(n_list, const1), 'c-',
             label= f"recursion fit: c={const1}")
    plt.plot(n_list, estimated_func_2(n_list, const2), 'k-',
             label= f"non-recursion fit: c={const2}")

    # Labeling the axis
    plt.xlabel('n')
    plt.ylabel('time')

    # Printing legend and constants and plot
    plt.legend()
    print("The constant of the recursion calculation is:")
    print("c:", const1)

    print("The constants of the non-recursion calculation are:")
    print("c:", const2)

    plt.show()

def estimated_func_2(input, constant):
    '''returns discrete points of estimated function'''
    input = np.array(input)
    return (constant*input)*np.log2(input)


######################################## HW number 3 ##############################################
######################
#  helper functions  #
######################

def verify_input_3(input_list):
    """
    Verifies that the input list contains only complex or real numbers.

    Args:
    input_list (numpy array or list): The list to be verified.

    """
    input_list = np.array(input_list)
    # check that the vector is iterable
    try:
        iter(input_list)
    except TypeError as ex:
        raise ValueError('Input vector is not iterable') from ex

        # check for wrong dtypes
    if not np.all((input_list.real + 1j * input_list.imag) == input_list):
        raise ValueError('Input vector contains variables that are not int,float or complex')

def convolve(arr_1,arr_2):
    '''convolve two arrays using convolution theorem'''
    x_dft = fft_1d_radix2(arr_1, 'FFTW_FORWARD')
    y_dft = fft_1d_radix2(arr_2,  'FFTW_FORWARD')
    mult = x_dft * y_dft
    conv = dft_1d(mult, 'FFTW_BACKWARD')
    return conv

######################
#       TASK 1       #
######################
def fft_1d(vec,direction):
    '''''
    compute the fft
    '''''
    # check the input vector
    verify_input_3(vec)
    vec_len = len(vec)
    if direction in ['FFTW_FORWARD','FFTW_BACKWARD']:
        pass
    else:
        raise ValueError('Invalid direction input')

    closet_power = 2
    while closet_power< 2*vec_len-1:
        closet_power *=2

    if direction == 'FFTW_FORWARD':
        x_array = vec * np.exp(1j * np.pi * (np.arange(vec_len) ** 2)/vec_len)
        y_array = np.exp(-(np.arange(vec_len) ** 2) * np.pi * 1j / vec_len)
        padded_x = np.pad(x_array, (0, closet_power - len(x_array)))
        zeros = np.array([0 for i in range(closet_power - 2 * vec_len + 1)])
        padded_y = np.concatenate((y_array, zeros, y_array[1:][::-1]))
        conv = convolve(padded_x,padded_y)[:vec_len]
        return ((y_array)**-1)*conv

    if direction == 'FFTW_BACKWARD':
        x_array = vec * np.exp(-1j * np.pi / vec_len * (np.arange(vec_len) ** 2))
        y_array = np.exp((np.arange(vec_len) ** 2) * np.pi * 1j / vec_len)
        padded_x = np.pad(x_array, (0, closet_power - len(x_array)))
        zeros = np.array([0 for i in range(closet_power - 2 * vec_len + 1)])
        padded_y = np.concatenate((y_array, zeros, y_array[1:][::-1]))
        conv = convolve(padded_x,padded_y)[:vec_len]
        return 1/vec_len*((y_array)**-1)*conv
    return "Wrong Direction"

###############
#   task 2    #
###############
def plot_fft_1d_timing():
    ''' plot time graph'''

    n_list =list(range(1,175))
    time_lst_for = np.zeros(len(n_list))
    time_lst_back = np.zeros(len(n_list))
    for i,arr_len in enumerate(n_list):
        vec = range(arr_len)
        start = time.time()
        fft_1d(vec,'FFTW_FORWARD')
        time_lst_for[i]=time.time()-start

        start = time.time()
        fft_1d(vec, 'FFTW_BACKWARD')
        time_lst_back[i] = time.time() - start


    # Draw the graph
    plt.plot(n_list, time_lst_for, 'b-', label='fft_1d forward run time')
    plt.plot(n_list, time_lst_back, 'k-', label='fft_1d backward run time')

    # Labeling the axis
    plt.xlabel('n')
    plt.ylabel('time')

    # Printing legend and constants and plot
    plt.legend()
    plt.show()

######################################## HW number 4 ##############################################

def verify_input_4(input_list):
    """
    Verifies that the input list contains only complex or real numbers.

    Args:
    input_list (numpy array or list): The list to be verified.

    """
    # check that the vector is iterable
    try:
        iter(input_list)
    except TypeError as ex:
        raise ValueError('Input vector is not iterable') from ex


    if len(input_list) %2 != 0:
        raise  ValueError('Array lenght should be even!')

    for item in input_list:
        # check for wrong dtypes
        if not isinstance(item, (complex, float, int)):
            raise ValueError('Input vector contains variables that are not int,float or complex')



def fft_1d_real(vec,direction):
    'calculate real fft (fft of real array)'
    verify_input_4(vec)
    vec_len = len(vec)
    h_arr =np.array(np.zeros(vec_len//2),dtype=complex)

    if direction == 'FFTW_FORWARD':
        for elem in vec:
            if not isinstance(elem, (float,int)):
                raise ValueError('Input vector should be real valued')
        const = np.exp(2*np.pi*1j/vec_len)
        for i in range(vec_len//2):
            h_arr[i] = vec[2*i]+1j*vec[2*i+1]
        h_arr = fft_1d(h_arr,direction)
        h_star = np.conjugate(h_arr)
        fft_arr = np.zeros(vec_len//2+1,dtype=complex)
        fft_arr[0] = np.sum(vec)
        fft_arr[vec_len // 2] = np.sum(vec[::2]) - np.sum(vec[1::2])
        for k in range(1,vec_len//2):
            fft_arr[k] = 1/2*(h_arr[k]+h_star[vec_len//2-k])\
                   -1j/2*(h_arr[k]-h_star[vec_len//2-k])*(const**k)
        fft_arr = np.concatenate((fft_arr,np.conjugate(fft_arr[1:vec_len//2][::-1])))
        return fft_arr

    if direction == 'FFTW_BACKWARD':
        const = np.exp(-2*np.pi*1j/vec_len)
        fft_arr = np.zeros(vec_len,dtype=complex)
        vec_conjugate = np.conjugate(vec)
        for k in range(vec_len//2):
            fft_arr[2*k] = 0.5*(vec[k]+vec_conjugate[vec_len//2-k])
            fft_arr[2*k+1] = 0.5*const**k*(vec[k]-vec_conjugate[vec_len//2-k])
        h_arr = fft_arr[::2] + np.multiply(1j,fft_arr[1::2])
        inverse = fft_1d(h_arr, direction)
        res = np.zeros(vec_len,dtype=complex)
        for i in range(vec_len//2):
            res[2*i] = np.real(inverse[i])
            res[2*i+1] = np.imag(inverse[i])
        return res
    return 'Wrong direction'


def plot_fft_1d_real_timing():
    ' plotting time of fft_1d real function'
    n_list = list(range(2, 200,2))
    time_lst_fft_1d = np.zeros(len(n_list))
    time_lst_ffd_1d_real = np.zeros(len(n_list))
    for i, arr_len in enumerate(n_list):
        vec = range(arr_len)
        start = time.time()
        fft_1d(vec, 'FFTW_FORWARD')
        time_lst_fft_1d[i] = time.time() - start

        start = time.time()
        fft_1d_real(vec, 'FFTW_FORWARD')
        time_lst_ffd_1d_real[i] = time.time() - start
        print(arr_len)

    # Draw the graph
    plt.plot(n_list, time_lst_fft_1d, 'b-', label='fft_1d run time')
    plt.plot(n_list, time_lst_ffd_1d_real, 'k-', label='fft_1d_real run time')

    # Labeling the axis
    plt.xlabel('n')
    plt.ylabel('time')

    # Printing legend and constants and plot
    plt.legend()
    plt.show()

######################################## HW number 5 ##############################################

def draw_graph(n_list,error,time_lst_for,time_lst_back,name):
    ' helper function to draw the graphs in task 2'

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('vec len')
    ax1.set_ylabel('error', color=color)
    ax1.plot(n_list, error, color=color, label= f'{name}_error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc=1)
    ax1.plot()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_1 = 'tab:blue'
    # Labeling the axis
    plt.xlabel('vec len')
    plt.ylabel('time')
    ax2.set_ylabel('time', color='k')  # we already handled the x-label with ax1
    ax2.plot(n_list, time_lst_for, color=color_1,label= f'{name}_time')
    ax2.plot(n_list, time_lst_back, color='k',label=f'{name}_time_fftw')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.legend(loc=0)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'comprasion of {name}')
    plt.show()

def fft_1d_fftw(vec,direction):
    'use pyfftw and numpy'
    verify_input_3(vec)
    arr_len = len(vec)
    if direction == 'FFTW_FORWARD':
        return pyfftw.interfaces.numpy_fft.ifft(vec)*arr_len
    return pyfftw.interfaces.numpy_fft.fft(vec)/arr_len

def fft_1d_real_fftw(vec,direction):
    'using fftw to compute real fft'
    verify_input(vec)
    arr_len = len(vec)
    if direction == 'FFTW_FORWARD':
        res =  pyfftw.interfaces.numpy_fft.rfft(vec)
        return np.concatenate((np.conjugate(res), res[1:-1][::-1]))
    return pyfftw.interfaces.numpy_fft.irfft(np.conjugate(vec[:arr_len//2+1]))

def compare_to_fftw():
    'comparing between fftw and my implementation'
    n_list = list(range(600,700,10))
    fft_1d_error_lst_for = []
    fft_1d_error_lst_back = []
    real_fft_1d_error_lst_for = []
    real_fft_1d_error_lst_back = []

    fft_1d_time_lst_for = []
    fftw_fft_1d_time_lst_for = []

    fft_1d_time_lst_back = []
    fftw_fft_1d_time_lst_back = []

    real_fft_1d_time_lst_for = []
    real_fftw_fft_1d_time_lst_for = []

    real_fft_1d_time_lst_back = []
    real_fftw_fft_1d_time_lst_back = []

    for arr_len in n_list:
        vec = range(arr_len)
        start_time = time.time()
        my_fft_for = fft_1d(vec,'FFTW_FORWARD')
        fft_1d_time_lst_for.append(time.time()-start_time)
        start_time = time.time()
        my_fft_back = fft_1d(vec, 'FFTW_BACKWARD')
        fft_1d_time_lst_back.append(time.time() - start_time)
        start_time = time.time()
        fftw_fft_for = fft_1d_fftw(vec,'FFTW_FORWARD')
        fftw_fft_1d_time_lst_for.append(time.time() - start_time)
        start_time = time.time()
        fftw_fft_back =fft_1d_fftw(vec,'FFTW_BACKWARD')
        fftw_fft_1d_time_lst_back.append(time.time() - start_time)
        start_time = time.time()
        my_fft_for_real = fft_1d_real(vec, 'FFTW_FORWARD')
        real_fft_1d_time_lst_for.append(time.time() - start_time)
        start_time = time.time()
        my_fft_back_real = fft_1d_real(my_fft_for_real, 'FFTW_BACKWARD')
        real_fft_1d_time_lst_back.append(time.time() - start_time)
        start_time = time.time()
        fftw_fft_for_real = fft_1d_real_fftw(vec, 'FFTW_FORWARD')
        real_fftw_fft_1d_time_lst_for.append(time.time() - start_time)
        start_time = time.time()
        fftw_fft_back_real = fft_1d_real_fftw(my_fft_for_real, 'FFTW_BACKWARD')
        real_fftw_fft_1d_time_lst_back.append(time.time() - start_time)

        fft_for_error = np.linalg.norm(my_fft_for-fftw_fft_for,ord=2)
        fft_back_error = np.linalg.norm(my_fft_back-fftw_fft_back,ord=2)
        fft_real_for_error = np.linalg.norm(my_fft_for_real-fftw_fft_for_real,ord=2)
        fft_real_back_error = np.linalg.norm(my_fft_back_real-fftw_fft_back_real,ord=2)

        fft_1d_error_lst_for.append(fft_for_error)
        fft_1d_error_lst_back.append(fft_back_error)
        real_fft_1d_error_lst_for.append\
            (fft_real_for_error)
        real_fft_1d_error_lst_back.append\
            (fft_real_back_error)

    draw_graph(n_list,fft_1d_error_lst_for,
                fft_1d_time_lst_for,fftw_fft_1d_time_lst_for,'fft_forward')
    draw_graph(n_list, fft_1d_error_lst_back,
               fft_1d_time_lst_back, fftw_fft_1d_time_lst_back,'fft_backward')

    draw_graph(n_list, real_fft_1d_error_lst_for,
               real_fft_1d_time_lst_for, real_fftw_fft_1d_time_lst_for,'real_fft_forward')
    draw_graph(n_list, real_fft_1d_error_lst_back,
               real_fft_1d_time_lst_back, real_fftw_fft_1d_time_lst_back,'real_fft_backward')

######################################## HW number 6 ##############################################

def fft_shift(arr):
    '''functuin to shift array so that the center will be at
    zero and symetric from negative to positive'''
    arr = np.array(arr)
    row_num, col_num = arr.shape
    return np.roll(arr,[row_num//2,col_num//2],[0,1])


def verify_input_2d(input_list):
    """
    Verifies that the input list contains only complex or real numbers.

    Args:
    input_list (numpy array or list): The list to be verified.(2D)

    """
    # check that the vector is iterable
    if not isinstance(input_list,np.ndarray) :
        input_list = np.array(input_list)
    try:
        iter(input_list)
    except TypeError as ex:
        raise ValueError('Input vector is not iterable') from ex
    for row in input_list:
        for item in row:
            # check for wrong dtypes
            if not isinstance(item, (complex, float, int,np.uint8,np.int32)):
                raise ValueError('Input vector contains variables that arent sint,float or complex')
    return input_list


def fft_2d(arr,direction):
    'implement 2D fft using 1D fft on each axis'
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)
    verify_input_2d(arr)
    row_num, col_num = arr.shape

    output_arr = np.zeros((row_num,col_num),dtype='complex')

    for row in range(row_num):
        output_arr[row,:] = fft_1d_fftw(arr[row,:],direction)

    for col in range(col_num):
        output_arr[:,col] = fft_1d_fftw(output_arr[:,col], direction)

    return output_arr

def cfft_2d(arr,direction):
    'implement 2D fft using 1D fft on each axis on a centred array'
    arr = fft_shift(arr)
    res = fft_2d(arr, direction)
    res = fft_shift(res)
    return res

def mask_generator(img,height,width,gap):
    'generate mask for noise remove'
    rows,columns = img.shape
    res = np.ones((rows,(columns-width)//2))
    for _ in range(width):
        mask_vec = np.ones((rows,1))
        for j in range(int(height)//2):
            mask_vec[rows//2+j+gap//2] = 0
            mask_vec[rows // 2 - j-gap//2] = 0
        res = np.concatenate((res,mask_vec),axis=1)
    res = np.concatenate((res, np.ones((rows,(columns-width)//2))),axis=1)
    return res

def check_best_filter_parameters(img_data):
    'find the best parameters of the mask'
    rows = img_data.shape[0]
    ratio_list =  np.arange(0.3, 0.8, 0.2)
    width_list = [5,7,9,11,13,15,17,19,21,23,25,27,29]

    for ratio in ratio_list:
        for gap in range(10, int(rows / 2 * ratio), 75):
            for width in width_list:
                mask = mask_generator(img_data,ratio*rows,width,gap)
                clean_moon_hat = cfft_2d(img_data, 'FFTW_FORWARD') * mask
                clean_moon = np.abs(cfft_2d(clean_moon_hat, 'FFTW_BACKWARD'))
                plt.imsave(f'tests/Clean_moon_{gap}_{width}_{ratio}.jpg',clean_moon)


######################################## HW number 7 #############################################
def shift_array(arr,shift):
    'helper function to shift 1D array'
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)

    vec_len = len(arr)
    arr = arr.reshape((1,vec_len))

    # Perform 2D complex Fast Fourier Transform in forward direction
    fft_mat = cfft_2d(arr,'FFTW_FORWARD')

    # Calculate the shifting factor using exponential function
    factor = np.exp(-2*np.pi*1j*np.arange(vec_len)*shift/vec_len).reshape(1,vec_len)

    # Apply the shifting factor to the Fourier transformed array
    fft_mat = fft_mat*factor

    # Perform 2D complex Fast Fourier Transform in backward direction
    fft_mat = cfft_2d(fft_mat,'FFTW_BACKWARD')

    return fft_mat

def pad_img(img):
    'padding image'
    img_pad = np.pad(img, img.shape)
    return img_pad

def shift_tester():
    'Test if the shift array works correctly using analytical calculations'

    arr_1 = [1,2,3,4,5,6,7,8,9]
    arr_1_shift  = [6,7,8,9,1,2,3,4,5]
    arr_2 = [1j,1,1+1j,2+1j,4,5]
    arr_2_shift = [4,5,1j,1,1+1j,2+1j]

    print(np.abs(shift_array(arr_1,4)))
    print((shift_array(arr_2, 2)))

    bool_1 = np.allclose(shift_array(arr_1,4), arr_1_shift)
    bool_2 = np.allclose(shift_array(arr_2,2), arr_2_shift)

    if (bool_1 is False or bool_2 is False):
        print('shift_array function ERROR')
    else:
        print("shift_array function works")

def rotate_90_deg_mul(image, angle):
    'manuel rotation of the image before rotate in angle less than 90'
    if angle not in {0, 90, 180, 270}:
        raise ValueError('angle is not one of {0, 90, 180, 270}')
    if angle == 0:
        return image
    if angle == 90:
        return image.T[::-1]
    if angle == 180:
        return image[::-1][:, ::-1]
    if angle == 270:
        return image.T[:, ::-1]
    return 'angle is not one of {0, 90, 180, 270}'
def rotate(image,angle,padded=False):
    'rotate image using 2D fft'
    image = verify_input_2d(image)
    # Pad the image based on the desired rotation angle
    if padded is False:
        image = pad_img(image)

    if angle > np.pi/2:
        angle_mod = angle - (angle%(np.pi/2))
        angle_mod = angle_mod/np.pi *180
        angle = angle%(np.pi/2)
        image =  rotate_90_deg_mul(image,angle_mod)

    angle= - angle

    rows,_ = image.shape

    # Perform row-wise shifting for the top half of the image
    for row in range(rows):
        image[row] = np.abs(shift_array(image[row],(row-rows/2)*np.tan(angle/2)))

    # Transpose the image and perform row-wise shifting for the bottom half of the image
    image = image.T
    for row in range(rows):
        image[row] = np.abs(shift_array(image[row],(row-rows/2)*np.tan(-angle/2)))
    image = image.T

    # Perform row-wise shifting for the entire image
    for row in range(rows):
        image[row] = np.abs(shift_array(image[row,:],(row-rows/2)*np.tan(angle/2)))

    return image


def pil_rotate_image(image_array, angle,padded=False):
    'rotate image using PIL'
    if padded is False:
        image_array = pad_img(image_array)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array)
    angle = angle/np.pi * 180
    # Rotate the image by the specified angle
    rotated_image = image.rotate(angle)

    # Convert the rotated image back to NumPy array
    rotated_array = np.array(rotated_image)

    return rotated_array

def task_2_compare(img,angle):
    "comparing PIL and FFT rotation"
    # Plot the image
    plt.imshow(img, cmap='gray')
    plt.show()

    # rotate the image with fft and PIL
    fft_rotation  = rotate(img,angle)
    pil_rotation = pil_rotate_image(img,angle)
    plt.imshow(np.hstack((fft_rotation, pil_rotation)), cmap='gray')
    plt.title('FFT rotation VS  Pil rotation')
    plt.show()

    fft_rotation_rec = rotate(fft_rotation.copy(),-angle,True)
    pil_rotation_rec = pil_rotate_image(pil_rotation.copy(),-angle,True)
    plt.imshow(np.hstack((fft_rotation_rec, pil_rotation_rec)), cmap='gray')
    plt.title('FFT Rec VS  Pil Rec')
    plt.show()
    padded_image = pad_img(img)
    diff_fft = fft_rotation_rec - padded_image
    diff_pil = pil_rotation_rec - padded_image

    fft_mse = ((diff_fft)**2).mean()
    pil_mse = ((diff_pil)**2).mean()

    fft_ssim = ssim(fft_rotation_rec,padded_image)
    pil_ssim = ssim(pil_rotation_rec,padded_image)

    plt.imshow(diff_fft,cmap='gray')
    plt.title('Difference between original and reconstruct with FFT rotation')
    plt.show()
    plt.imshow(diff_pil,cmap='gray')
    plt.title('Difference between original and reconstruct with PIL rotation')
    plt.show()

    print(f'PIl MSE: {pil_mse}')
    print(f'PIl SSIM: {pil_ssim}')
    print(f'FFT MSE: {fft_mse}')
    print(f'FFT SSIM: {fft_ssim}')


######################################## HW number 8 #############################################


def dct_matrix(size):
    'create the dct matrix'
    dct_mat = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            dct_mat[i,j] = np.pi*(i+1/2)*j/size
    dct_mat = np.cos(dct_mat)
    return dct_mat.T

def dct_2d(arr, direction):
    'implement 2d DCT using matrix mult'
    dct_mat = dct_matrix(8)
    if direction == 'FORWARD':
        res = np.matmul(np.matmul(dct_mat,arr),dct_mat.T)
        return res
    if direction == 'BACKWARD':
        dct_mat = np.linalg.inv(dct_matrix(8))
        res = np.matmul(np.matmul(dct_mat,arr), dct_mat.T)
        return res
    return None

def split_into_blocks(array):
    'divied array to 8 by 8 sub arrays'
    blocks = []
    rows, cols = array.shape
    # Split the array into 8x8 blocks
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            block = array[i:i+8, j:j+8]
            block = block - 128.0
            blocks.append(block)
    return blocks

def pad_image_for_compression(array):
    'pad to the closet multiply by 8'

    rows,columns = array.shape
    array = np.pad(array,(((8-rows)%8,0),((8-columns)%8,0)))
    return array

def zigzag(block):
    """
    implement zigzag pattern for jpeg compression
    """
    indexes_tuples = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    res = []
    for index in indexes_tuples:
        res.append(block[index[0]][index[1]])
    return res

def reconstruct_image(blocks,img_shape):
    """
    Reconstructs the original image from a list of 8x8 blocks.
    """
    image = np.zeros(img_shape) # Initialize an empty image
    cnt = 0
    rows,columns = img_shape
    for i in range(0, rows, 8):
        for j in range(0, columns,8):
                image[i: i + 8, j: j + 8] = blocks[cnt]
                cnt += 1
    return image

def reverse_zigzag(zigzag_vector):
    'return original array from zigzag vectors'
    block = np.zeros((8,8))  # Initialize an empty block
    zigzag_order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]  # Zigzag traversal order

    # Reconstruct the block using the zigzag vector
    for i, value in enumerate(zigzag_vector):
        row, column = zigzag_order[i]
        block[row][column] = value
    return block



def encode(input_image_filename, output_compressed_filename,quality):
    '''encoder function:
    image ---> compressed image
    using DCT
    '''
    q_mat = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])
    if quality < 50:
        s_const = 5000 / quality
    else:
        s_const = 200 - 2 * quality
    if quality == 100:
        q_hat = np.ones(q_mat.shape)
    else:
        q_hat = np.floor((s_const * q_mat + 50) / 100)

    img = plt.imread(input_image_filename)
    img = np.array(img)
    rows,columns = img.shape

    # Check if the array can be divided into 8x8 blocks
    if rows % 8 != 0 or columns % 8 != 0:
        img = pad_image_for_compression(img)

    blocks_lst = split_into_blocks(img)

    zigzag_block_lst = []
    for block in blocks_lst:
        dct_block = np.floor(dct_2d(block,'FORWARD')//q_hat).astype(np.int16)
        zigzag_block_lst.append(zigzag(dct_block))
    zigzag_block_lst = np.asarray(zigzag_block_lst)
    compressed_zigzag_block_lst = zlib.compress(zigzag_block_lst.tobytes())
    img_shape_coded = img.shape[0].to_bytes(2,'little') + img.shape[1].to_bytes(2,'little')

    compressed_lst_and_quality = bytes([quality]) + img_shape_coded + compressed_zigzag_block_lst

    with open(output_compressed_filename, 'wb') as file:
        file.write(compressed_lst_and_quality)
    return output_compressed_filename

def decompress_vectors(compressed_data,vector_size):
    'decompress compressed list of vectors'
    # Decompress the compressed data using zlib
    decompressed_data = zlib.decompress(compressed_data)

    #Convert the decompressed data to a NumPy array
    vectors = np.frombuffer(decompressed_data, dtype=np.int16)

    #Determine the size of each vector and reshape the array accordingly
    vector_length = len(vectors) // vector_size  # Assuming all vectors have the same size
    vectors = vectors.reshape(vector_length, vector_size)

    #Return the decompressed vectors
    return vectors

def decode(input_compressed_filename, output_image_filename):
    '''decoder function:
    compressed image---> image
    using DCT
    '''

    with open(input_compressed_filename, 'rb') as file:
        encoded_file = file.read()
    quality = encoded_file[0]

    shape_bytes = encoded_file[1:5]
    img_shape = int.from_bytes(shape_bytes[:2],'little'), int.from_bytes(shape_bytes[2:],'little')
    compressed_zigzag_vec = decompress_vectors(encoded_file[5:],64)
    quality = int(quality)

    q_mat = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    if quality < 50:
        s_const = 5000 / quality
    else:
        s_const = 200 - 2 * quality
    if quality == 100:
        q_hat = np.ones(q_mat.shape)
    else:
        q_hat = np.floor((s_const * q_mat + 50) / 100)
    res = []
    for row in range(compressed_zigzag_vec.shape[0]):
        block = np.floor(reverse_zigzag(compressed_zigzag_vec[row,:])*q_hat)
        block = dct_2d(block,'BACKWARD')
        block = block + 128
        res.append(block)
    img = reconstruct_image(res,img_shape)

    image = Image.fromarray(img)
    image.save(output_image_filename)

############################# Task 2 ###############################
def compress_decompress(input_file_name,intermedian_file_name,output_file_name,quality):
    'helper function that compress and then decompress given image'
    encode(input_file_name,intermedian_file_name,quality)
    decode(intermedian_file_name,output_file_name)


def plot_images():
    'task 2 function - plot the images and data about the compression'
    images_names = \
        ['project8_images/1','project8_images/3','project8_images/12','project8_images/47']

    for image_name in images_names:
        original_image = np.array(Image.open(image_name + '.gif'))
        img_shape = original_image.shape
        original_size = os.path.getsize(image_name + '.gif')
        psnr_values = []
        compression_ratios = []

        for quality in [5,10,25,50]:
            compress_decompress\
                (image_name + '.gif', image_name + '_inter', image_name + '_reconstruction.gif', quality)
            reconstructed_image = \
                np.array(Image.open(image_name + '_reconstruction.gif'))

            # Calculate PSNR
            mse = np.mean((np.array(original_image) - np.array(reconstructed_image)) ** 2)
            psnr = round(20 * np.log10(255 / (mse**0.5)),1)
            psnr_values.append(psnr)

            # Calculate compression ratio
            compressed_size = os.path.getsize(image_name+'_inter')
            compression_ratio = round((compressed_size / original_size) * 100,1)
            compression_ratios.append(compression_ratio)


            # Display the images
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.text(0,img_shape[1]+50,f"PSNR (Quality: {quality}): {psnr}")

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image)
            plt.title(f"Reconstructed Image (Quality: {quality})")
            plt.text(0,img_shape[1]+50
                     ,f"Compression Ratio (Quality: {quality}): {compression_ratio}%")
        plt.show()

def main():
    'main function to revice the arguments for the compression'
    parser = argparse.ArgumentParser(description="JPEG Image Encoding and Decoding")
    parser.add_argument("mode", choices=["encode", "decode"], help="Mode: 'encode' or 'decode'")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--quality", type=int, default=100, help="Image quality (1-100, default: 100)")
    args = parser.parse_args()
    if args.mode == "encode":
        encode(args.input, args.output, args.quality)
        print("Image encoded successfully.")

    elif args.mode == "decode":
        if not os.path.isfile(args.output):
            print('Argument Error: <compressed_file> does not exist')
            return None
        decode(args.input, args.output)
        print("Image decoded successfully.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()