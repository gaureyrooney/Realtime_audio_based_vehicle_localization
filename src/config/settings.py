# settings.py

VM_B1_DEVICE_INDEX = 37  # update with your sounddevice indexes
VM_B2_DEVICE_INDEX = 34
VM_B3_DEVICE_INDEX = 36

SAMPLE_RATE = 48000
DO_CALIBRATION = True

# If calibration finds offset with absolute value > CLAMP_OFFSET, we clamp it
CLAMP_OFFSET = 8192
MAX_WINDOW_SIZE = 16384
MAX_HOP_LENGTH = 8192