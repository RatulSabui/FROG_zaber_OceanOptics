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
from scipy.interpolate import interp1d


############################################################################################################################################

folder = 'D:/allCodes/zaber_FROG/trial_data8'

filename = 'full_spectrum_data.npz'

approx_pulse_width = 400 #fs

padding_thickness = 0.05 # 0.1 means 10%
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


def background_subtract(Isig, freq_sub=0, delay_sub=0):
    """
    Subtract background from a 2D intensity matrix.

    Parameters:
        Isig (np.ndarray): 2D intensity matrix (shape: wavelength x delay)
        freq_sub (float): subtraction factor for frequency background (along rows)
        delay_sub (float): subtraction factor for delay background (along columns)

    Returns:
        np.ndarray: background-subtracted intensity matrix
    """
    A = Isig.copy()

    if freq_sub != 0:
        # Minimum along delay axis for each frequency (row)
        min_freq = np.min(A, axis=1, keepdims=True)
        A = A - freq_sub * min_freq

    if delay_sub != 0:
        # Minimum along frequency axis for each delay (column)
        min_delay = np.min(A, axis=0, keepdims=True)
        A = A - delay_sub * min_delay

    return A


def enforce_nonnegativity(Isig):
    """
    Replace negative values in the intensity matrix with zero.

    Parameters:
        Isig (np.ndarray): 2D intensity matrix

    Returns:
        np.ndarray: intensity matrix with negatives clipped to zero
    """
    return np.clip(Isig, 0, None)

def bin_to_uniform_grid(Isig, Tau, Lam, N_tau_out, N_lam_out):
    """
    Interpolate Isig intensity matrix onto uniform delay and wavelength axes.

    Parameters:
        Isig (np.ndarray): 2D intensity matrix (shape: len(Lam) x len(Tau))
        Tau (np.ndarray): original delay axis (length matches Isig shape)
        Lam (np.ndarray): original wavelength axis (length matches Isig shape)
        N_tau_out (int): number of output delay points
        N_lam_out (int): number of output wavelength points

    Returns:
        Isig_binned (np.ndarray): resampled intensity matrix (N_lam_out x N_tau_out)
        Tau_new (np.ndarray): uniform delay axis
        Lam_new (np.ndarray): uniform wavelength axis
    """
    # Create new uniform axes spanning the original data ranges
    Tau_new = np.linspace(Tau.min(), Tau.max(), N_tau_out)
    Lam_new = np.linspace(Lam.min(), Lam.max(), N_lam_out)

    # Create output coordinate grid for interpolation
    Tau_grid, Lam_grid = np.meshgrid(Tau_new, Lam_new)

    # Define interpolator (Lam along axis 0, Tau along axis 1 in Isig)
    interp_func = RegularGridInterpolator((Lam, Tau), Isig, bounds_error=False, fill_value=0)

    # Prepare points for interpolation - flatten grids and stack
    points = np.array([Lam_grid.ravel(), Tau_grid.ravel()]).T

    # Interpolate values on new uniform grid
    Isig_binned = interp_func(points).reshape(N_lam_out, N_tau_out)

    return Isig_binned, Tau_new, Lam_new


def wavelength_to_frequency(Lam, Isig, c=299792458):
    """
    Convert wavelength axis (Lam, in nm) to frequency axis (Hz), adjust intensity matrix accordingly.

    Parameters:
        Lam (np.ndarray): 1D wavelength array in nanometers, assumed sorted ascending
        Isig (np.ndarray): 2D intensity matrix (shape: len(Lam) x len(delay))
        c (float): speed of light in m/s, default 299792458

    Returns:
        freq (np.ndarray): frequency axis in Hz, sorted ascending
        Isig_freq (np.ndarray): intensity matrix reordered to match ascending freq axis
    """
    # Convert wavelength from nm to meters
    Lam_m = Lam * 1e-9

    # Compute frequency (Hz)
    freq = c / Lam_m

    # Reverse to ascending frequency because freq decreases as wavelength increases
    freq = freq[::-1]

    # Reverse intensity matrix rows accordingly
    Isig_freq = Isig[::-1, :]

    return freq, Isig_freq


def next_power_of_two(n):
    """ Return the next power of two greater than or equal to n """
    return 1 << (n - 1).bit_length()

def pad_matrix_and_extend_axes(matrix, freq, delay, extra_x=0, extra_y=0, enforce_power_of_two=True):
    """
    Pad a 2D matrix with zeros and extend freq and delay axes by linear extrapolation.

    Parameters:
        matrix (np.ndarray): 2D array (freq x delay)
        freq (np.ndarray): 1D frequency axis array (length matches matrix.shape[0])
        delay (np.ndarray): 1D delay axis array (length matches matrix.shape[1])
        extra_x (int): additional rows to pad (frequency axis)
        extra_y (int): additional cols to pad (delay axis)
        enforce_power_of_two (bool): pad output sizes to next power of two

    Returns:
        padded_matrix (np.ndarray): zero-padded matrix
        freq_extended (np.ndarray): extended frequency axis
        delay_extended (np.ndarray): extended delay axis
    """
    x, y = matrix.shape
    target_x = x + extra_x
    target_y = y + extra_y

    if enforce_power_of_two:
        target_x = next_power_of_two(target_x)
        target_y = next_power_of_two(target_y)

    pad_x = target_x - x
    pad_y = target_y - y

    # Padding sizes on each side (center input)
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top

    # Pad matrix
    padded_matrix = np.pad(matrix,
                           pad_width=((pad_left, pad_right), (pad_top, pad_bottom)),
                           mode='constant', constant_values=0)

    # Extend frequency axis (assumes freq sorted ascending)
    freq_spacing_start = freq[1] - freq[0]
    freq_spacing_end = freq[-1] - freq[-2]

    freq_extended_start = freq[0] - np.arange(pad_left, 0, -1) * freq_spacing_start
    freq_extended_end = freq[-1] + np.arange(1, pad_right + 1) * freq_spacing_end
    freq_extended = np.concatenate((freq_extended_start, freq, freq_extended_end))

    # Extend delay axis (assumes delay sorted ascending)
    delay_spacing_start = delay[1] - delay[0]
    delay_spacing_end = delay[-1] - delay[-2]

    delay_extended_start = delay[0] - np.arange(pad_top, 0, -1) * delay_spacing_start
    delay_extended_end = delay[-1] + np.arange(1, pad_bottom + 1) * delay_spacing_end
    delay_extended = np.concatenate((delay_extended_start, delay, delay_extended_end))

    return padded_matrix, freq_extended, delay_extended


def intensity_to_amplitude(intensity_matrix):
    """
    Converts a 2D intensity matrix to amplitude by taking the square root.
    
    Parameters:
        intensity_matrix (np.ndarray): 2D array of intensity values (non-negative).
    
    Returns:
        amplitude_matrix (np.ndarray): 2D array of amplitude values.
    """
    # Ensure no negative values before taking sqrt, clip to zero
    intensity_nonneg = np.clip(intensity_matrix, 0, None)
    
    amplitude_matrix = np.sqrt(intensity_nonneg)
    
    return amplitude_matrix

def save_to_frg(filename, intensity_matrix, temporal_calibration, central_wavelength, spectral_calibration):
    """
    Save the FROG trace intensity matrix to .frg file format.
    
    Parameters:
        filename (str): Path to save the .frg file.
        intensity_matrix (np.ndarray): 2D intensity matrix (wavelength x delay).
        temporal_calibration (float): Temporal calibration in fs/pixel or appropriate units.
        central_wavelength (float): Central wavelength in nm.
        spectral_calibration (float): Spectral calibration in nm/pixel.
        
    File format:
        1st line: width height temporal_calibration spectral_calibration central_wavelength
        followed by lines of intensity values for each row (wavelength)
    """
    height, width = intensity_matrix.shape
    with open(filename, 'w') as f:
        header_line = f"{width}\t{height}\t{temporal_calibration}\t{spectral_calibration}\t{central_wavelength}\n"
        f.write(header_line)
        
        # Write each row of the matrix as tab-separated values
        for row in intensity_matrix:
            row_str = '\t'.join(f"{val:.6e}" for val in row)
            f.write(row_str + '\n')

def linearize_intensity(wavelengths, intensity_matrix):
    """
    Linearize the intensity matrix along the spectral axis from wavelength to frequency linear spacing.
    
    Parameters:
        wavelengths (np.ndarray): 1D array of wavelengths (nm), assumed sorted ascending but not necessarily linear.
        intensity_matrix (np.ndarray): 2D intensity matrix with shape (len(wavelengths), N_delay).
    
    Returns:
        linear_freqs (np.ndarray): 1D array of linearly spaced frequencies (Hz).
        linear_intensity (np.ndarray): Intensity matrix re-interpolated to linear frequency grid (same shape).
    """

    c = 299792458  # speed of light in m/s
    # Convert wavelengths (nm) to frequency (Hz)
    freqs = c / (wavelengths * 1e-9) 
    
    # Create linearly spaced frequency grid from min to max
    linear_freqs = np.linspace(freqs.min(), freqs.max(), len(freqs))
    
    # Interpolate intensity matrix spectral axis to linear_freqs
    interp_func = interp1d(freqs[::-1], intensity_matrix[::-1, :], axis=0, kind='linear', bounds_error=False, fill_value=0)
    linear_intensity = interp_func(linear_freqs)
    
    return linear_freqs, linear_intensity


def linearize_intensity_wavelength(wavelengths, intensity_matrix):
    """
    Linearize the intensity matrix along the spectral axis to a linear wavelength grid.
    
    Parameters:
        wavelengths (np.ndarray): 1D array of wavelengths (nm), assumed sorted ascending but not necessarily linear.
        intensity_matrix (np.ndarray): 2D intensity matrix with shape (len(wavelengths), N_delay).
    
    Returns:
        linear_wavelengths (np.ndarray): 1D array of linearly spaced wavelengths (nm).
        linear_intensity (np.ndarray): Intensity matrix re-interpolated to linear wavelength grid (same shape).
    """
    # Create linearly spaced wavelength grid from min to max
    linear_wavelengths = np.linspace(wavelengths.min(), wavelengths.max(), len(wavelengths))
    
    # Interpolate intensity matrix spectral axis to linear_wavelengths
    interp_func = interp1d(wavelengths, intensity_matrix, axis=0, kind='linear', bounds_error=False, fill_value=0)
    linear_intensity = interp_func(linear_wavelengths)
    
    return linear_wavelengths, linear_intensity



def plot_central_slices(delay_array, wavelength_array, intensity_matrix, save_dir):
    """
    Plots and saves line plots of central row (intensity vs delay) and
    central column (intensity vs wavelength) from the 2D intensity matrix.
    
    Parameters:
        delay_array (np.ndarray): 1D array of delay values (x-axis for row plot)
        wavelength_array (np.ndarray): 1D array of wavelength values (x-axis for column plot)
        intensity_matrix (np.ndarray): 2D intensity matrix (wavelength x delay)
        save_dir (str): directory path to save the plots
    
    Saves plots as 'central_row_vs_delay.png' and 'central_column_vs_wavelength.png' in save_dir.
    """
    
    # Central row index (wavelength axis)
    central_row_idx = intensity_matrix.shape[0] // 2
    central_row_intensity = intensity_matrix[central_row_idx, :]
    
    plt.figure()
    plt.plot(delay_array, central_row_intensity, '-b')
    plt.xlabel('Delay')
    plt.ylabel('Intensity')
    plt.title('Central Row Intensity vs Delay')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/central_row_vs_delay.png')
    plt.close()
    
    # Central column index (delay axis)
    central_col_idx = intensity_matrix.shape[1] // 2
    central_col_intensity = intensity_matrix[:, central_col_idx]
    
    plt.figure()
    plt.plot(wavelength_array, central_col_intensity, '-r')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Central Column Intensity vs Wavelength')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/central_column_vs_wavelength.png')
    plt.close()
    

#####################################################################################################################

orig_size, pad_side = calculate_equal_square_padding(total_dimension, padding_thickness)


intensity_matrix, wavelengths, delay_array, distances = read_npz_file(file_path)
print("Intensity matrix shape:", intensity_matrix.shape)
print("Initial Wavelength array length:", len(wavelengths))
print("Initial Delays array length:", len(delay_array))
print(wavelengths)

# --- Step 2: Center the delay axis so peak is at zero ---
centered_delay_array, peak_delay, peak_col_idx = center_delay_axis_by_peak(intensity_matrix, np.array(delay_array))

centered_op = save_path +"/"+ 'centered_matrix.png'
plot_and_save_intensity_colormesh(centered_delay_array, wavelengths,intensity_matrix, centered_op, title="Intensity(Peak Centered)", cmap='viridis')

#doing normal background subtraction and removing negatrive values
Isig_temp = background_subtract(intensity_matrix, freq_sub=1.0, delay_sub=1.0)
intensity_matrix = enforce_nonnegativity(Isig_temp)

print(wavelengths)

# since the wavelength axis does not have linear values
wavelengths, intensity_matrix = linearize_intensity_wavelength(wavelengths, intensity_matrix)

print("After linearization.")
print(wavelengths)

#crops the matrix with relevant delay values
cropped_delays, cropped_matrix = crop_matrix_by_delay_range(centered_delay_array, intensity_matrix, selection_range=approx_pulse_width)

#crops the matrix and makes it a square matrix
cropped_wavelengths, cropped_matrix = crop_matrix_to_square_by_max_row_sum(cropped_matrix, wavelengths)


width, height, temp_calib, spec_calib, center_wl = get_matrix_properties(cropped_delays, cropped_wavelengths)

print("After Cropping.")
print(f"Width (x-axis length): {width}")
print(f"Height (y-axis length): {height}")
print(f"Temporal calibration (x resolution): {temp_calib}")
print(f"Spectral calibration (y resolution): {spec_calib}")
print(f"Central wavelength: {center_wl}")

cropped_op = save_path +"/"+ 'cropped_matrix.png'
plot_and_save_intensity_colormesh(cropped_delays, cropped_wavelengths,cropped_matrix, cropped_op, title="Intensity(cropped)", cmap='viridis')


matrix_uniform, delay_uniform, wavelength_uniform = bin_to_uniform_grid(cropped_matrix, cropped_delays, cropped_wavelengths, orig_size, orig_size)

freq, Isig_freq = wavelength_to_frequency(wavelength_uniform, matrix_uniform)

padded_mat, freq_ext, delay_ext = pad_matrix_and_extend_axes(Isig_freq, freq, delay_uniform, extra_x=pad_side, extra_y=pad_side,enforce_power_of_two=True)

amp_matrix = intensity_to_amplitude(padded_mat)

amp_mat_op = save_path +"/"+ 'amp_matrix.png'
plot_and_save_intensity_colormesh(delay_ext, freq_ext, padded_mat, amp_mat_op, title="Intensity(cropped)", cmap='viridis')

frg_file_path = save_path +"/"+ 'frog_trace.frg'
save_to_frg(frg_file_path, padded_mat, temp_calib, center_wl, spec_calib)

# plotting linecuts 
plot_central_slices(delay_ext, freq_ext, padded_mat, save_path)

width, height, temp_calib, spec_calib, center_wl = get_matrix_properties(delay_ext, freq_ext)

print("After Saving to frg.")
print(f"Width (x-axis length): {width}")
print(f"Height (y-axis length): {height}")
print(f"Temporal calibration (x resolution): {temp_calib}")
print(f"Spectral calibration (y resolution): {spec_calib}")
print(f"Central wavelength: {center_wl}")