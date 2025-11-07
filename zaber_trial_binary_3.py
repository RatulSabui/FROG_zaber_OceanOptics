from zaber_motion import Library, Units, LogOutputMode
from zaber_motion.binary import Connection

# Turn off verbose logging
Library.set_log_output(LogOutputMode.OFF)

PORT_NAME = "COM3"
BAUD_RATE = 9600
DEVICE_ADDRESS = 2   # From your detection output
AXIS_NUMBER = 1

try:
    with Connection.open_serial_port(PORT_NAME, baud_rate=BAUD_RATE) as connection:
        print(f"✅ Connected to {PORT_NAME} at {BAUD_RATE} baud.")

        devices = connection.detect_devices()
        print("Detected devices:", devices)

        device = connection.get_device(DEVICE_ADDRESS)

        print("Homing...")
        device.home()
        device.wait_until_idle()
        print("✅ Homed.")

        pos = device.get_position(Units.LENGTH_MILLIMETRES)
        print(f"Current position: {pos:.4f} mm")

        print("Moving to 5 mm...")
        device.move_absolute(5, Units.LENGTH_MILLIMETRES)
        device.wait_until_idle()

        pos = device.get_position(Units.LENGTH_MILLIMETRES)
        print(f"Now at {pos:.4f} mm")

        print("✅ Done.")

except Exception as e:
    print(f"❌ Error: {e}")
