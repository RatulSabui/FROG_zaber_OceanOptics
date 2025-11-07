"""_summary_
    creates a lineplot for the intensity corresponding to a particular wavelength
    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_delay_from_filename(filename):
    pattern = r'(-?\d+\.\d+)fs'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Delay info not found in filename: {filename}")

def find_intensity_for_wavelength(wavelength, data_dir):
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    
    data = []

    for fname in csv_files:
        delay = extract_delay_from_filename(fname)
        df = pd.read_csv(fname)
        # Find intensity for wavelength closest to the target
        df['wl_diff'] = (df.iloc[:,0] - wavelength).abs()
        closest_row = df.loc[df['wl_diff'].idxmin()]
        intensity = closest_row.iloc[1]  # intensity column
        data.append((delay, intensity))

    df_data = pd.DataFrame(data, columns=['Delay_fs', 'Intensity'])
    df_data = df_data.sort_values('Delay_fs').reset_index(drop=True)

    # Save to CSV
    csv_save_path = os.path.join(data_dir, f'intensity_vs_delay_{wavelength}nm.csv')
    df_data.to_csv(csv_save_path, index=False)

    # Plotting
    plt.figure()
    plt.plot(df_data['Delay_fs'], df_data['Intensity'], marker='o')
    plt.xlabel('Delay (fs)')
    plt.ylabel(f'Intensity at {wavelength} nm (a.u.)')
    plt.title(f'Intensity vs Delay at {wavelength} nm')
    pic_path = os.path.join(data_dir, f'intensity_vs_delay_{wavelength}nm.png')
    plt.savefig(pic_path)
    plt.close()

    print(f"Saved sorted data CSV: {csv_save_path}")
    print(f"Saved plot PNG: {pic_path}")

    return df_data



data_dir = 'D:/allCodes/zaber_FROG/trial_data11'
target_wavelength = 520.0  # in nm
df_intensity = find_intensity_for_wavelength(target_wavelength, data_dir)
