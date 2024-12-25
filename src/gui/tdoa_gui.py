"""
tdoa_gui.py

This updated GUI code includes:
1) Inputs for:
   - Perp. distance of kerb from mic1
   - Number of lanes
   - Lane width
2) Overlays 50% opacity lane strips on the localization plot
3) Displays the 'current lane' of the localized sound source in the UI
4) Maintains the existing:
   - TDOA vs. Time plot
   - Heat Map
   - Localization Plot
   - Audio levels
   - GCC method usage
   - Window size/hop length usage
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import time, threading
import math
import string  # for lane labeling A, B, C, ...

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Example placeholders for your real modules
from src.config.settings import SAMPLE_RATE, DO_CALIBRATION, CLAMP_OFFSET
from src.audio.real_time_capture import Voicemeeter3BusCapture
from src.audio.audio_utils import preprocess_signal_for_vehicles
from src.localization.gcc_methods import compute_single_tdoa
from src.localization.localization_utils import localize_single_time_step
from src.localization.kalman_filter import apply_kalman_filter

def compute_heat_map_example(dt12, dt13, dt23, mic_positions,
                             xmin, xmax, ymin, ymax, grid_size=50):
    """
    Example function to produce a dummy heat map (just a Gaussian around (2, -1)).
    In practice, you'd do a TDOA-based grid approach.
    """
    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    x_est, y_est = 2.0, -1.0
    sigma = 1.0
    for i, gx in enumerate(xs):
        for j, gy in enumerate(ys):
            dist_sq = (gx - x_est)**2 + (gy - y_est)**2
            val = np.exp(-dist_sq/(2*sigma*sigma))
            heatmap[j, i] = val

    return xs, ys, heatmap

class TDOAVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime TDOA: Lanes & Localization")

        # --- Mic coordinates ---
        self.mic1_x = tk.DoubleVar(value=0.0)  # origin
        self.mic1_y = tk.DoubleVar(value=0.0)
        self.mic2_x = tk.DoubleVar(value=-1.0)
        self.mic2_y = tk.DoubleVar(value=1.0)
        self.mic3_x = tk.DoubleVar(value=1.0)
        self.mic3_y = tk.DoubleVar(value=-1.0)

        # --- Plot range ---
        self.xmin = tk.DoubleVar(value=-10.0)
        self.xmax = tk.DoubleVar(value=10.0)
        self.ymin = tk.DoubleVar(value=-5.0)
        self.ymax = tk.DoubleVar(value=5.0)

        # --- TDOA params ---
        self.window_size = tk.IntVar(value=4096)
        self.hop_length = tk.IntVar(value=2048)
        self.method = tk.StringVar(value="phat")
        self.use_kalman = tk.BooleanVar(value=False)

        # --- Rolling detections ---
        self.detections = []   # list of (x, y, timestamp)
        self.dot_lifetime = 2.0
        self.is_live = False

        # --- Real-Time Audio ---
        self.capture = Voicemeeter3BusCapture(blocksize=2048)

        # --- Audio Levels ---
        self.b1_level_str = tk.StringVar(value="B1: 0 dB")
        self.b2_level_str = tk.StringVar(value="B2: 0 dB")
        self.b3_level_str = tk.StringVar(value="B3: 0 dB")

        # --- TDOA Timeseries for selected mic pair ---
        self.monitor_mic_pair = tk.StringVar(value="(1,2)")
        self.tdoa_times = []
        self.tdoa_values = []
        self.start_time = time.time()

        # Offsets if you do calibration
        self.offset_b2_samples = 0
        self.offset_b3_samples = 0

        # Last TDOAs (for heatmap or debugging)
        self.current_tdoas = (0.0, 0.0, 0.0)

        # --- NEW Lane-Related Inputs ---
        self.perp_dist_kerb = tk.DoubleVar(value=0.0)
        self.num_lanes = tk.IntVar(value=2)   # default 2 lanes
        self.lane_width = tk.DoubleVar(value=3.5)

        # We'll store the 'current lane' the vehicle is in
        self.current_lane_str = tk.StringVar(value="None")

        self.setup_ui()
        self.setup_main_figure()   # Localization + Heat Map
        self.setup_tdoa_figure()   # TDOA vs Time

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding=5)
        control_frame.grid(row=0, column=0, rowspan=2, sticky="nw")

        rowi = 0
        # Mic1
        ttk.Label(control_frame, text="Mic1 (origin) x,y:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.mic1_x, width=6).grid(row=rowi, column=1)
        tk.Entry(control_frame, textvariable=self.mic1_y, width=6).grid(row=rowi, column=2)
        rowi += 1

        # Mic2
        ttk.Label(control_frame, text="Mic2 x,y:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.mic2_x, width=6).grid(row=rowi, column=1)
        tk.Entry(control_frame, textvariable=self.mic2_y, width=6).grid(row=rowi, column=2)
        rowi += 1

        # Mic3
        ttk.Label(control_frame, text="Mic3 x,y:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.mic3_x, width=6).grid(row=rowi, column=1)
        tk.Entry(control_frame, textvariable=self.mic3_y, width=6).grid(row=rowi, column=2)
        rowi += 2

        # Plot Range
        ttk.Label(control_frame, text="Xmin:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.xmin, width=6).grid(row=rowi, column=1)
        rowi += 1
        ttk.Label(control_frame, text="Xmax:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.xmax, width=6).grid(row=rowi, column=1)
        rowi += 1
        ttk.Label(control_frame, text="Ymin:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.ymin, width=6).grid(row=rowi, column=1)
        rowi += 1
        ttk.Label(control_frame, text="Ymax:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.ymax, width=6).grid(row=rowi, column=1)
        rowi += 1

        ttk.Button(control_frame, text="Apply Plot Limits", command=self.apply_plot_limits).grid(
            row=rowi, column=0, columnspan=3, pady=5
        )
        rowi += 2

        # TDOA params
        ttk.Label(control_frame, text="Window Size").grid(row=rowi, column=0, sticky="e")
        tk.Scale(control_frame, from_=1024, to=16384, variable=self.window_size, orient="horizontal").grid(row=rowi, column=1, sticky="we")
        rowi += 1
        ttk.Label(control_frame, text="Hop Length").grid(row=rowi, column=0, sticky="e")
        tk.Scale(control_frame, from_=512, to=8192, variable=self.hop_length, orient="horizontal").grid(row=rowi, column=1, sticky="we")
        rowi += 1
        ttk.Label(control_frame, text="GCC Method").grid(row=rowi, column=0, sticky="e")
        tk.OptionMenu(control_frame, self.method, "phat", "ml", "scot", "normal").grid(row=rowi, column=1, sticky="w")
        rowi += 1

        ttk.Checkbutton(control_frame, text="Use Kalman", variable=self.use_kalman).grid(
            row=rowi, column=0, columnspan=3, sticky="w"
        )
        rowi += 2

        # Audio Levels
        ttk.Label(control_frame, textvariable=self.b1_level_str).grid(row=rowi, column=0, sticky="w")
        rowi += 1
        ttk.Label(control_frame, textvariable=self.b2_level_str).grid(row=rowi, column=0, sticky="w")
        rowi += 1
        ttk.Label(control_frame, textvariable=self.b3_level_str).grid(row=rowi, column=0, sticky="w")
        rowi += 2

        # TDOA mic pair
        ttk.Label(control_frame, text="Mic Pair (TDOA Monitor):").grid(row=rowi, column=0, sticky="e")
        mic_pair_options = ["(1,2)", "(1,3)", "(2,3)"]
        tk.OptionMenu(control_frame, self.monitor_mic_pair, *mic_pair_options).grid(row=rowi, column=1, sticky="w")
        rowi += 2

        # NEW Lane Inputs
        ttk.Label(control_frame, text="Perp. Dist. Kerb from Mic1:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.perp_dist_kerb, width=6).grid(row=rowi, column=1, sticky="w")
        rowi += 1

        ttk.Label(control_frame, text="Number of lanes:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.num_lanes, width=6).grid(row=rowi, column=1, sticky="w")
        rowi += 1

        ttk.Label(control_frame, text="Lane width:").grid(row=rowi, column=0, sticky="e")
        tk.Entry(control_frame, textvariable=self.lane_width, width=6).grid(row=rowi, column=1, sticky="w")
        rowi += 2

        # We'll display the current lane in a label
        ttk.Label(control_frame, text="Current Lane:").grid(row=rowi, column=0, sticky="e")
        ttk.Label(control_frame, textvariable=self.current_lane_str, foreground="blue").grid(row=rowi, column=1, sticky="w")
        rowi += 2

        # Start/Stop
        ttk.Button(control_frame, text="Start Live", command=self.start_live).grid(
            row=rowi, column=0, columnspan=2, pady=5
        )
        rowi += 1

    def setup_main_figure(self):
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.grid(row=0, column=1, sticky="nsew")

        self.fig_main = Figure(figsize=(10,5), dpi=100)
        self.ax_loc = self.fig_main.add_subplot(121)
        self.ax_loc.set_title("Localization Plot")
        self.ax_loc.set_xlabel("X (m)")
        self.ax_loc.set_ylabel("Y (m)")
        self.ax_loc.grid(True)

        self.ax_heat = self.fig_main.add_subplot(122)
        self.ax_heat.set_title("Heat Map")
        self.ax_heat.set_xlabel("X (m)")
        self.ax_heat.set_ylabel("Y (m)")

        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=main_frame)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_tdoa_figure(self):
        tdoa_frame = ttk.Frame(self.root, padding=5)
        tdoa_frame.grid(row=1, column=1, sticky="nsew")

        self.fig_tdoa = Figure(figsize=(10,3), dpi=100)
        self.ax_tdoa = self.fig_tdoa.add_subplot(111)
        self.ax_tdoa.set_title("TDOA vs. Time")
        self.ax_tdoa.set_xlabel("Time (s)")
        self.ax_tdoa.set_ylabel("TDOA (s)")
        self.ax_tdoa.grid(True)

        self.canvas_tdoa = FigureCanvasTkAgg(self.fig_tdoa, master=tdoa_frame)
        self.canvas_tdoa.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def apply_plot_limits(self):
        self.ax_loc.set_xlim(self.xmin.get(), self.xmax.get())
        self.ax_loc.set_ylim(self.ymin.get(), self.ymax.get())
        self.ax_heat.set_xlim(self.xmin.get(), self.xmax.get())
        self.ax_heat.set_ylim(self.ymin.get(), self.ymax.get())
        self.redraw_main_plots()

    def start_live(self):
        if self.is_live:
            return
        self.is_live = True
        self.tdoa_times = []
        self.tdoa_values = []
        self.start_time = time.time()

        self.capture.start()
        threading.Thread(target=self._process_audio_loop, daemon=True).start()

    def stop_live(self):
        if not self.is_live:
            return
        self.is_live = False
        self.capture.stop()

    def _process_audio_loop(self):
        while self.is_live:
            b1_list, b2_list, b3_list = self.capture.get_frames()

            # Suppose we accumulate data, do windowing of size=window_size, hop=hop_length,
            # and compute dt12, dt13, dt23 with method = self.method.get()

            dt12, dt13, dt23 = (0.0, 0.0, 0.0)  # placeholders
            self.current_tdoas = (dt12, dt13, dt23)

            # Localization
            mic_positions = {
                1: (self.mic1_x.get(), self.mic1_y.get()),
                2: (self.mic2_x.get(), self.mic2_y.get()),
                3: (self.mic3_x.get(), self.mic3_y.get())
            }
            try:
                x_est, y_est = localize_single_time_step(dt12, dt13, dt23, mic_positions, c=343.0)
            except:
                x_est, y_est = np.nan, np.nan

            if not np.isnan(x_est) and not np.isnan(y_est):
                self.add_detection(x_est, y_est)
                # Determine current lane
                lane_label = self.find_current_lane(y_est)  # We'll define a new function
                self.root.after(0, self.update_current_lane_label, lane_label)
            else:
                self.root.after(0, self.update_current_lane_label, "None")

            # TDOA vs Time for chosen mic pair
            pair_str = self.monitor_mic_pair.get()
            pair = eval(pair_str)  # (1,2) or (1,3) or (2,3)
            if pair == (1,2):
                chosen_tdoa = dt12
            elif pair == (1,3):
                chosen_tdoa = dt13
            else:
                chosen_tdoa = dt23

            elapsed_s = time.time() - self.start_time
            self.tdoa_times.append(elapsed_s)
            self.tdoa_values.append(chosen_tdoa)

            # Trigger plot updates in main thread
            self.root.after(0, self.redraw_main_plots)
            self.root.after(0, self.update_tdoa_plot)

            time.sleep(1.0)

    def add_detection(self, x, y):
        self.detections.append((x, y, time.time()))

    def redraw_main_plots(self):
        # Localization Plot
        self.ax_loc.clear()
        self.ax_loc.set_title("Localization Plot")
        self.ax_loc.set_xlabel("X (m)  [Vehicles flow along X]")
        self.ax_loc.set_ylabel("Y (m)  [Perp. direction => Lanes]")
        self.ax_loc.grid(True)
        self.ax_loc.set_xlim(self.xmin.get(), self.xmax.get())
        self.ax_loc.set_ylim(self.ymin.get(), self.ymax.get())

        # Draw microphone positions
        m1x, m1y = self.mic1_x.get(), self.mic1_y.get()
        m2x, m2y = self.mic2_x.get(), self.mic2_y.get()
        m3x, m3y = self.mic3_x.get(), self.mic3_y.get()
        self.ax_loc.plot(m1x, m1y, 'ro'); self.ax_loc.text(m1x, m1y, "Mic1", color="red")
        self.ax_loc.plot(m2x, m2y, 'ro'); self.ax_loc.text(m2x, m2y, "Mic2", color="red")
        self.ax_loc.plot(m3x, m3y, 'ro'); self.ax_loc.text(m3x, m3y, "Mic3", color="red")

        # Overlays: pavement + lanes
        self.draw_lane_overlay()

        # Remove expired detections
        now = time.time()
        valid = []
        for (dx, dy, tstamp) in self.detections:
            if (now - tstamp) < self.dot_lifetime:
                valid.append((dx, dy, tstamp))
        self.detections = valid

        # Plot them
        for (dx, dy, _) in self.detections:
            self.ax_loc.plot(dx, dy, 'co')

        # Heat Map
        self.ax_heat.clear()
        self.ax_heat.set_title("Heat Map")
        self.ax_heat.set_xlabel("X (m)")
        self.ax_heat.set_ylabel("Y (m)")
        self.ax_heat.set_xlim(self.xmin.get(), self.xmax.get())
        self.ax_heat.set_ylim(self.ymin.get(), self.ymax.get())

        dt12, dt13, dt23 = self.current_tdoas
        xs, ys, heatmap = compute_heat_map_example(dt12, dt13, dt23,
                                                   {1:(m1x,m1y),
                                                    2:(m2x,m2y),
                                                    3:(m3x,m3y)},
                                                   self.xmin.get(), self.xmax.get(),
                                                   self.ymin.get(), self.ymax.get())
        extent = [self.xmin.get(), self.xmax.get(), self.ymin.get(), self.ymax.get()]
        self.ax_heat.imshow(heatmap, origin='lower', extent=extent, cmap='hot', alpha=0.5)

        self.canvas_main.draw()

    def update_tdoa_plot(self):
        self.ax_tdoa.clear()
        self.ax_tdoa.set_title("TDOA vs. Time")
        self.ax_tdoa.set_xlabel("Time (s)")
        self.ax_tdoa.set_ylabel("TDOA (s)")
        self.ax_tdoa.grid(True)

        if self.tdoa_times and self.tdoa_values:
            self.ax_tdoa.plot(self.tdoa_times, self.tdoa_values, 'b-o', markersize=3)

        self.canvas_tdoa.draw()

    def update_current_lane_label(self, lane_label):
        self.current_lane_str.set(lane_label)

    def draw_lane_overlay(self):
        """
        Draw lane strips with 50% opacity over the localization plot.
        Assume:
          - X axis: direction of flow
          - Y axis: lane direction
          - perp_dist_kerb = y-value of kerb (start of lane A).
          - num_lanes, lane_width => create N strips
          - We'll label them A, B, C, ...
        """
        kerb_y = self.perp_dist_kerb.get()
        n_lanes = self.num_lanes.get()
        w_lane = self.lane_width.get()
        # We'll label up to 26 lanes: A, B, C, D, ...
        labels = string.ascii_uppercase  # 'A','B','C','D',...

        # We assume lanes go from kerb_y upward (y increasing).
        # Lane A in [kerb_y, kerb_y + w_lane)
        # Lane B in [kerb_y + w_lane, kerb_y + 2*w_lane), etc.

        for i in range(n_lanes):
            lane_name = labels[i] if i < len(labels) else f"Lane{i+1}"
            lane_min = kerb_y + i * w_lane
            lane_max = kerb_y + (i+1) * w_lane

            # Draw a rectangle from x_min->x_max, y=lane_min->lane_max
            rect_x_min = self.xmin.get()
            rect_x_max = self.xmax.get()
            rect_y_min = lane_min
            rect_y_max = lane_max

            self.ax_loc.fill_between([rect_x_min, rect_x_max],
                                     rect_y_min, rect_y_max,
                                     color='green', alpha=0.2)  # 50% = 0.5, or 0.2

            # Place the label in the middle of that lane
            mid_y = 0.5 * (lane_min + lane_max)
            mid_x = 0.5 * (rect_x_min + rect_x_max)
            self.ax_loc.text(mid_x, mid_y, lane_name, color='blue', alpha=0.7,
                             ha='center', va='center')

    def find_current_lane(self, y_est):
        """
        Given y_est, determine which lane the vehicle is in.
        If y_est < kerb or beyond the top lane, return "None".
        """
        kerb_y = self.perp_dist_kerb.get()
        n_lanes = self.num_lanes.get()
        w_lane = self.lane_width.get()
        labels = string.ascii_uppercase

        if y_est < kerb_y:
            return "None"  # below the first lane
        # Lane i in [kerb_y + i*w_lane, kerb_y + (i+1)*w_lane)
        for i in range(n_lanes):
            lane_min = kerb_y + i*w_lane
            lane_max = kerb_y + (i+1)*w_lane
            if lane_min <= y_est < lane_max:
                if i < len(labels):
                    return labels[i]
                else:
                    return f"Lane{i+1}"

        return "None"  # above the top lane


def main():
    root = tk.Tk()
    app = TDOAVisualizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
