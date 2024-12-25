# kalman_filter.py

import numpy as np

def apply_kalman_filter(positions):
    if not positions:
        return positions

    dt = 0.1
    x = np.array([positions[0][0], positions[0][1], 0.0, 0.0])
    P = np.eye(4)*1.0
    Q = np.eye(4)*0.1
    R = np.eye(2)*0.5

    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    filtered = []
    for pos in positions:
        z = np.array([pos[0], pos[1]], dtype=float)

        x = F @ x
        P = F @ P @ F.T + Q

        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        filtered.append((x[0], x[1]))

    return filtered
