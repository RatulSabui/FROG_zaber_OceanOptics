"""_summary_
    once the matrix has been created, 
    
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d, maximum_filter1d


############################################################################################################################################

folder = 'D:/allCodes/zaber_FROG/trial_data8'

filename = 'full_spectrum_data.npz'

approx_pulse_width = 150 #fs

padding_thickness = 0.1 # 0.1 means 10%
total_dimension = 512 # total binned matrix has to be order of 2

#for saving the outputs
save_path = folder + "/" + 'output'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
file_path = folder + "/" + filename








"""
save_processed_data_npy(delays, wavelengths, intensity_matrix, 'output_file.npy')
plot_and_save_intensity_colormesh(delays, wavelengths, intensity_matrix, 'output_zoomed.png', y_range=(490, 530))
"""

############################################################################################################################################

def read_npz_file(filename):
    npzfile = np.load(filename)
    intensity_matrix = npzfile['intensity_matrix']
    wavelengths = npzfile['wavelengths']
    delays_fs = npzfile['delays_fs']
    distances = npzfile['distances']
    return intensity_matrix, wavelengths, delays_fs, distances


def calculate_equal_square_padding(target_size, padding_fraction):
    """
    Calculate original square matrix size and equal padding thickness on all four sides.

    Parameters:
        target_size (int): Final size of square matrix after padding (e.g., 512)
        padding_fraction (float): Fraction of target size for total padding (e.g., 0.1 for 10%)

    Returns:
        original_size (int): Size of matrix before padding (rounded)
        pad_each_side (int): Padding thickness on each of the four sides (equal)
    """
    total_padding = round(target_size * padding_fraction)

    # Make total padding divisible by 4, adjusting to nearest multiple
    total_padding_adjusted = 4 * round(total_padding / 4)

    original_size = target_size - total_padding_adjusted
    pad_each_side = total_padding_adjusted // 4

    return original_size, pad_each_side


def plot_and_save_intensity_colormesh(delays, wavelengths, counts_matrix, save_path, title="Intensity Colormesh", cmap='viridis',
                                      x_range=None, y_range=None):
    """
    Plot the counts matrix as a colormesh image and save it as a PNG file without displaying it.
    Allows specifying x and y axis ranges for zooming into a subsection of the data.
    
    Parameters:
    - delays: 1D array of delay values (x-axis)
    - wavelengths: 1D array of wavelength values (y-axis)
    - counts_matrix: 2D array of averaged intensity/counts (rows = wavelengths, cols = delays)
    - save_path: filepath where to save the PNG image
    - title: title of the plot
    - cmap: colormap for the image
    - x_range: tuple (xmin, xmax) to specify delay axis limits, default None uses full range
    - y_range: tuple (ymin, ymax) to specify wavelength axis limits, default None uses full range
    """
    plt.figure(figsize=(8, 6))
    
    delay_edges = np.linspace(delays[0], delays[-1], len(delays)+1)
    wavelength_edges = np.linspace(wavelengths[0], wavelengths[-1], len(wavelengths)+1)
    
    mesh = plt.pcolormesh(delay_edges, wavelength_edges, counts_matrix, shading='auto', cmap=cmap)
    
    plt.xlabel('Delay (s)')
    plt.ylabel('Wavelength')
    plt.title(title)
    plt.colorbar(mesh, label='Averaged Counts')
    
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def center_delay_axis_by_peak(counts_matrix, delay_array):
    """
    Shifts the delay axis so the peak column (with maximum summed counts) is at zero.

    Parameters:
    - counts_matrix: 2D numpy array (wavelengths x delays)
    - delay_array: 1D numpy array of delay values (length = number of columns in counts_matrix)

    Returns:
    - new_delay_array: 1D array with zero at peak column
    - peak_delay: original delay value where peak occurs
    - peak_col_index: column index of the peak
    """
    # Sum counts column-wise (sum across all wavelengths for each delay)
    column_sums = counts_matrix.sum(axis=0)

    # Find index of maximum sum (peak)
    peak_col_index = np.argmax(column_sums)
    peak_delay = delay_array[peak_col_index]

    # Shift delay axis so that peak_delay is now zero
    new_delay_array = delay_array - peak_delay

    return new_delay_array, peak_delay, peak_col_index

def crop_matrix_by_delay_range(centered_delay_array, intensity_matrix, selection_range):
    """
    Crops the intensity matrix to only include columns where the centered delay is within ±selection_range.

    Parameters:
    - centered_delay_array: 1D numpy array of centered delay values (zero at peak)
    - intensity_matrix: 2D numpy array of counts (rows = wavelengths, cols = delays)
    - selection_range: float, maximum absolute delay value to include (in seconds)

    Returns:
    - cropped_delays: 1D numpy array of delays within the selected range
    - cropped_matrix: 2D numpy array with columns within the selected range
    """
    mask = np.abs(centered_delay_array) <= selection_range
    cropped_delays = centered_delay_array[mask]
    cropped_matrix = intensity_matrix[:, mask]
    return cropped_delays, cropped_matrix


def crop_matrix_to_square_by_max_row_sum(intensity_matrix, y_axis_array):
    """
    Crops the intensity matrix and y-axis array to produce a square matrix based on the row with maximum row-sum.

    Parameters:
    - intensity_matrix: 2D numpy array (rows = wavelengths, cols = delays)
    - y_axis_array: 1D numpy array corresponding to rows of intensity_matrix (e.g., wavelengths)

    Returns:
    - cropped_y_axis: 1D numpy array cropped to match square matrix rows
    - cropped_matrix: 2D square numpy array cropped around the row with maximum row-sum
    """
    num_cols = intensity_matrix.shape[1]  # Number of columns (delays)
    row_sums = intensity_matrix.sum(axis=1)  # sum over columns for each row
    max_row_idx = np.argmax(row_sums)

    half_size = num_cols // 2

    # Determine row start and end indices, ensuring bounds are respected
    start_row = max(0, max_row_idx - half_size)
    end_row = start_row + num_cols

    # Adjust if end_row exceeds total number of rows
    if end_row > intensity_matrix.shape[0]:
        end_row = intensity_matrix.shape[0]
        start_row = max(0, end_row - num_cols)

    # Crop rows of matrix and corresponding y-axis values
    cropped_matrix = intensity_matrix[start_row:end_row, :]
    cropped_y_axis = y_axis_array[start_row:end_row]

    return cropped_y_axis, cropped_matrix


def get_matrix_properties(x_axis_array, y_axis_array):
    """
    Calculate properties of the matrix based on x and y axes arrays.

    Parameters:
    - x_axis_array: 1D numpy array representing the x-axis values (e.g., delays)
    - y_axis_array: 1D numpy array representing the y-axis values (e.g., wavelengths)

    Returns:
    - width: int, number of elements along x-axis (columns)
    - height: int, number of elements along y-axis (rows)
    - temporal_calibration: float, resolution along x-axis (average spacing between consecutive x values)
    - spectral_calibration: float, resolution along y-axis (average spacing between consecutive y values)
    - central_wavelength: float, y-axis value at center row of the matrix
    """
    width = len(x_axis_array)
    height = len(y_axis_array)

    # Calculate temporal calibration (x-axis resolution) as mean difference between adjacent values
    if width > 1:
        temporal_calibration = (x_axis_array[-1] - x_axis_array[0]) / (width - 1)
    else:
        temporal_calibration = 0.0

    # Calculate spectral calibration (y-axis resolution) as mean difference between adjacent values
    if height > 1:
        spectral_calibration = (y_axis_array[-1] - y_axis_array[0]) / (height - 1)
    else:
        spectral_calibration = 0.0

    # Find central wavelength (middle element of y-axis array)
    central_idx = height // 2
    central_wavelength = y_axis_array[central_idx]

    return width, height, temporal_calibration, spectral_calibration, central_wavelength


#####################################################################################################################

orig_size, pad_side = calculate_equal_square_padding(total_dimension, padding_thickness)


intensity_matrix, wavelengths, delay_array, distances = read_npz_file(file_path)
print("Intensity matrix shape:", intensity_matrix.shape)
print("Initial Wavelength array length:", len(wavelengths))
print("Initial Delays array length:", len(delay_array))

# --- Step 2: Center the delay axis so peak is at zero ---
centered_delay_array, peak_delay, peak_col_idx = center_delay_axis_by_peak(intensity_matrix, np.array(delay_array))

centered_op = save_path +"/"+ 'centered_matrix.png'
plot_and_save_intensity_colormesh(centered_delay_array, wavelengths,intensity_matrix, centered_op, title="Intensity(Peak Centered)", cmap='viridis')


#crops the matrix with relevant delay values
cropped_delays, cropped_matrix = crop_matrix_by_delay_range(centered_delay_array, intensity_matrix, selection_range=approx_pulse_width)

#crops the matrix and makes it a square matrix
cropped_wavelengths, cropped_square_matrix = crop_matrix_to_square_by_max_row_sum(cropped_matrix, wavelengths)

width, height, temp_calib, spec_calib, center_wl = get_matrix_properties(cropped_delays, cropped_wavelengths)

print(f"Width (x-axis length): {width}")
print(f"Height (y-axis length): {height}")
print(f"Temporal calibration (x resolution): {temp_calib}")
print(f"Spectral calibration (y resolution): {spec_calib}")
print(f"Central wavelength: {center_wl}")

cropped_op = save_path +"/"+ 'cropped_matrix.png'
plot_and_save_intensity_colormesh(cropped_delays, cropped_wavelengths,cropped_square_matrix, cropped_op, title="Intensity(cropped)", cmap='viridis')