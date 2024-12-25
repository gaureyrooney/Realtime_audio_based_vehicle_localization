# real_time_capture.py

import sounddevice as sd
import numpy as np
import threading
from collections import deque

from src.config.settings import (
    VM_B1_DEVICE_INDEX,
    VM_B2_DEVICE_INDEX,
    VM_B3_DEVICE_INDEX,
    SAMPLE_RATE
)

class Voicemeeter3BusCapture:
    """
    Opens 3 InputStreams (B1, B2, B3), each with a queue of (timestamp, mono_data).
    """

    def __init__(self, blocksize=2048, max_queue_len=50):
        self.blocksize = blocksize
        self.sample_rate = SAMPLE_RATE
        self.max_queue_len = max_queue_len

        self.queues = {
            "B1": deque(maxlen=max_queue_len),
            "B2": deque(maxlen=max_queue_len),
            "B3": deque(maxlen=max_queue_len)
        }
        self.lock = threading.Lock()

        self.is_running = False
        self.streams = {}

    def _callback_b1(self, indata, frames, time_info, status):
        if status:
            print("[B1] status:", status)
        device_timestamp = time_info.inputBufferAdcTime
        mono_data = indata[:, 0].copy()
        with self.lock:
            self.queues["B1"].append((device_timestamp, mono_data))

    def _callback_b2(self, indata, frames, time_info, status):
        if status:
            print("[B2] status:", status)
        device_timestamp = time_info.inputBufferAdcTime
        mono_data = indata[:, 0].copy()
        with self.lock:
            self.queues["B2"].append((device_timestamp, mono_data))

    def _callback_b3(self, indata, frames, time_info, status):
        if status:
            print("[B3] status:", status)
        device_timestamp = time_info.inputBufferAdcTime
        mono_data = indata[:, 0].copy()
        with self.lock:
            self.queues["B3"].append((device_timestamp, mono_data))

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        try:
            self.streams["B1"] = sd.InputStream(
                device=VM_B1_DEVICE_INDEX,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.blocksize,
                callback=self._callback_b1
            )
            self.streams["B1"].start()

            self.streams["B2"] = sd.InputStream(
                device=VM_B2_DEVICE_INDEX,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.blocksize,
                callback=self._callback_b2
            )
            self.streams["B2"].start()

            self.streams["B3"] = sd.InputStream(
                device=VM_B3_DEVICE_INDEX,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.blocksize,
                callback=self._callback_b3
            )
            self.streams["B3"].start()

            print("[Voicemeeter3BusCapture] Streams started.")
        except Exception as e:
            print("[Voicemeeter3BusCapture] Error starting streams:", e)
            self.is_running = False

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False

        for k in ["B1", "B2", "B3"]:
            if k in self.streams and self.streams[k] is not None:
                self.streams[k].stop()
                self.streams[k].close()
                self.streams[k] = None
        print("[Voicemeeter3BusCapture] Streams stopped.")

    def get_frames(self):
        """Return all (timestamp, samples) from each queue, then clear them."""
        with self.lock:
            b1 = list(self.queues["B1"])
            b2 = list(self.queues["B2"])
            b3 = list(self.queues["B3"])
            self.queues["B1"].clear()
            self.queues["B2"].clear()
            self.queues["B3"].clear()
        return b1, b2, b3
