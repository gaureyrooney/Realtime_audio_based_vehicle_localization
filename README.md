# Real-Time Vehicle Localization and Lane Occupancy Detection

This project implements a real-time microphone array system designed to detect vehicle positions and determine lane occupancy using Time Difference of Arrival (TDOA) principles and Generalized Cross-Correlation (GCC) methods.

## Features

- **Real-Time Localization**: Tracks vehicle positions using a 3-microphone array.
- **Lane Occupancy Detection**: Identifies the lane occupied by the detected vehicle.
- **Visualization**:
  - **Localization Plot**: Displays real-time vehicle positions with lane overlays.
  - **Heatmap**: Shows probable vehicle locations.
  - **TDOA vs. Time Plot**: Illustrates time delay changes for selected microphone pairs.
- **User Inputs**:
  - Perpendicular distance from the kerb to microphone 1.
  - Number of lanes (labeled A, B, C, etc.).
  - Lane width.

## Theoretical Background

### Time Difference of Arrival (TDOA)

TDOA is a technique that estimates the position of a sound source by measuring the difference in arrival times of the sound at multiple spatially separated sensors. The time differences correspond to distance differences, which can be used to determine the source's location. :contentReference[oaicite:0]{index=0}

### Localization via Hyperbola Intersection

In a 2D plane, the locus of points with a constant difference in distances to two fixed points (microphones) forms a hyperbola. By measuring TDOA between multiple pairs of microphones, hyperbolic curves are established. The intersection of these hyperbolas pinpoints the sound source's location. :contentReference[oaicite:1]{index=1}

### Generalized Cross-Correlation (GCC) Methods

GCC methods enhance TDOA estimation by applying frequency domain weighting to the cross-correlation of signals received at different microphones. Common GCC weighting functions include:

- **PHAT (Phase Transform)**: Emphasizes phase information, improving robustness in reverberant environments. :contentReference[oaicite:2]{index=2}
- **SCOT (Smoothed Coherence Transform)**: Balances magnitude and phase information.
- **ML (Maximum Likelihood)**: Maximizes the likelihood function under specific noise assumptions.

## Lane Occupancy Determination

By establishing a coordinate system with microphone 1 at the origin, the X-axis aligned with vehicle flow, and the Y-axis perpendicular to it, the system calculates the vehicle's position. Knowing the perpendicular distance from the kerb, lane width, and number of lanes, the Y-coordinate of the vehicle's position determines its lane occupancy.

## How to Use

1. **Install Dependencies**: Ensure Python 3.8+ is installed. Install required libraries using:
   ```bash
   pip install -r requirements.txt

## Configure Microphones

Connect three microphones and route them to Voicemeeter buses:

- Microphone 1 → B1
- Microphone 2 → B2
- Microphone 3 → B3

## Run the Application

## Input Parameters via GUI

- **Microphone Coordinates**: Enter coordinates for Mic 1, Mic 2, and Mic 3.
- **Plot Bounds**: Define Xmin, Xmax, Ymin, and Ymax for visualization limits.
- **Lane Configuration**:
  - Perpendicular Distance of Kerb from Mic1.
  - Number of Lanes.
  - Lane Width.
- **TDOA Parameters**:
  - Window Size.
  - Hop Length.
  - GCC Method (PHAT, ML, SCOT, Normal).

## Visualization Outputs

- **Localization Plot**: Real-time vehicle position with lane overlays.
- **Heatmap**: Probable vehicle locations.
- **TDOA vs. Time Plot**: Time delay changes for selected microphone pairs.
