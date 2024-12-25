# localization_utils.py

import numpy as np
from scipy.optimize import least_squares

def localization_residuals(vars, mic_positions, c, dt12, dt13, dt23):
    X, Y = vars
    x1, y1 = mic_positions[1]
    x2, y2 = mic_positions[2]
    x3, y3 = mic_positions[3]

    d1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
    d2 = np.sqrt((X - x2)**2 + (Y - y2)**2)
    d3 = np.sqrt((X - x3)**2 + (Y - y3)**2)

    r1 = (d1 - d2) - c*dt12
    r2 = (d1 - d3) - c*dt13
    r3 = (d2 - d3) - c*dt23
    return np.array([r1, r2, r3])

def localize_single_time_step(dt12, dt13, dt23, mic_positions, c=343.0, initial_guess=(0,0)):
    result = least_squares(localization_residuals,
                           initial_guess,
                           args=(mic_positions, c, dt12, dt13, dt23),
                           method='lm')
    X_est, Y_est = result.x
    return X_est, Y_est
