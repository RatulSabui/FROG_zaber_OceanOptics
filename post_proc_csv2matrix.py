# After the data has been captured by the stage/spectrometer operations, this code can be used toi create the matrix
# Its functionality has also been included in the main code zaber_frog_trial_3
# do to be fair this code is redundant


##################################################################################################

SAVE_DIR = "D:/allCodes/zaber_FROG/trial_data8"  # Folder to save spectra
CENTRAL_WAVELENGTH = 519                          # nanometers - central wavelength
BANDWIDTH = 80                                    # crop bandwidth in nanometers
CROP_TIME_RANGE = [-500, 500]                     # crop time range in fs
OUTPUT_DIR = SAVE_DIR                             


##################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.constants as const
import os
import csv
import pandas as pd

def calc_dist_from_time(pulse_width=400):
    """Calculate distance from pulse width (fs → mm)."""
    dist = pulse_width * const.femto * const.c / const.milli
    return dist


def calc_time_from_dist(dist=2):
    """Calculate time (fs) from distance (mm)."""
    time = dist * const.milli / const.c / const.femto
    return time

def process_spectra(data_dir, central_wavelength, bandwidth, crop_time_range_fs, output_dir):
    """
    Loads CSV spectra files with headers, constructs intensity matrix, crops, saves as numpy and images.
    Creates 2 plots: one with delay (fs) x-axis and another with distance (same units as calc_dist_from_time output).
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Now processing the csv files.")

    crop_wavelength_range = [central_wavelength - bandwidth / 2, central_wavelength + bandwidth / 2]
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    delays = []
    all_intensities = []

    # Iterate over files and collect delays and sorted intensities
    for fname in csv_files:
        base = os.path.basename(fname)
        parts = base.split("_")
        delay_str = parts[2].replace("fs", "")
        delay_fs = float(delay_str)
        delays.append(delay_fs)

        df = pd.read_csv(fname)
        # Sort the data by wavelength column explicitly
        df_sorted = df.sort_values("Wavelength (nm)").reset_index(drop=True)
        all_intensities.append(df_sorted["Intensity (a.u.)"].values)

    delays = np.array(delays)
    distances_sorted = np.array([calc_dist_from_time(d) for d in delays])

    # Use the sorted wavelength axis from the first file
    df_first = pd.read_csv(csv_files[0])
    wavelengths = np.sort(df_first["Wavelength (nm)"].values)

    # Stack columns (each file is one column)
    intensity_matrix = np.column_stack(all_intensities)

    # Sort columns and delay/distances arrays by the delay values
    sorted_indices = np.argsort(delays)
    delays_sorted = delays[sorted_indices]
    distances_sorted = distances_sorted[sorted_indices]
    intensity_matrix = intensity_matrix[:, sorted_indices]

    # Save full data
    np.save(os.path.join(output_dir, "full_intensity_matrix.npy"), intensity_matrix)
    np.save(os.path.join(output_dir, "axes.npy"), {"wavelengths": wavelengths, "delays_fs": delays_sorted})
    np.savez_compressed(
        os.path.join(output_dir, "full_spectrum_data.npz"),
        intensity_matrix=intensity_matrix,
        wavelengths=wavelengths,
        delays_fs=delays_sorted,
        distances=distances_sorted,
    )

    # Crop wavelength
    wave_mask = (wavelengths >= crop_wavelength_range[0]) & (wavelengths <= crop_wavelength_range[1])
    cropped_waves = wavelengths[wave_mask]
    cropped_matrix = intensity_matrix[wave_mask, :]

    # Crop delay/time range
    time_mask = (delays_sorted >= crop_time_range_fs[0]) & (delays_sorted <= crop_time_range_fs[1])
    cropped_delays = delays_sorted[time_mask]
    cropped_distances = distances_sorted[time_mask]
    cropped_matrix = cropped_matrix[:, time_mask]

    np.save(os.path.join(output_dir, "cropped_intensity_matrix.npy"), cropped_matrix)

    # Plot 1: delay (fs) vs wavelength
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(cropped_delays, cropped_waves, cropped_matrix, shading="auto")
    plt.colorbar(label="Intensity (a.u.)")
    plt.xlabel("Delay (fs)")
    plt.ylabel("Wavelength (nm)")
    plt.title("Cropped Intensity Map - Time Domain")
    plt.savefig(os.path.join(output_dir, "colormap_delay.png"))
    plt.close()

    # Plot 2: distance vs wavelength
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(cropped_distances, cropped_waves, cropped_matrix, shading="auto")
    plt.colorbar(label="Intensity (a.u.)")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Wavelength (nm)")
    plt.title("Cropped Intensity Map - Distance Domain")
    plt.savefig(os.path.join(output_dir, "colormap_distance.png"))
    plt.close()
    
    
process_spectra(
        data_dir=SAVE_DIR,
        central_wavelength=CENTRAL_WAVELENGTH,
        bandwidth=BANDWIDTH,
        crop_time_range_fs=CROP_TIME_RANGE,
        output_dir=SAVE_DIR,
    )