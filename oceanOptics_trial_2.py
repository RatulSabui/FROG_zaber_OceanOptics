import seabreeze


# Select backend before importing spectrometers
seabreeze.use('pyseabreeze')  # or 'cseabreeze'

#print(seabreeze.get_backend())


from seabreeze.spectrometers import list_devices, Spectrometer

spec = Spectrometer.from_first_available()

devices = list_devices()
print("Devices found: ", devices)

print("Model:", spec.model)
print("Serial number:", spec.serial_number)
print("Integration time range:", spec.integration_time_micros_limits)

# Set integration time (e.g. 100 ms)
spec.integration_time_micros(100_000)

# Acquire one spectrum
wavelengths = spec.wavelengths()
intensities = spec.intensities()

for wl, it in zip(wavelengths[:5], intensities[:5]):
    print(f"{wl:.2f} nm → {it:.1f}")
"""


if not devices:
    print("No spectrometer detected.")
else:
    for i, dev in enumerate(devices):
        print(f"Device {i}: {dev}")
       
    # Open first device
    spec = Spectrometer.from_first_available()
    print(f"Using device: {spec}")
"""

