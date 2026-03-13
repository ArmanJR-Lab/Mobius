# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
"""
Lateral steering controller for the comma.ai controls challenge simulator.

Interface
---------
The simulator calls `controller.update(...)` every step (10 Hz) and expects
a scalar steering command in return.

    def update(self, target_lataccel, current_lataccel, state, future_plan) -> float

Arguments
---------
target_lataccel : float
    Desired lateral acceleration for this timestep.

current_lataccel : float
    Actual lateral acceleration from the previous timestep (model output).

state : State namedtuple
    Current vehicle state with fields:
      - roll_lataccel (float): lateral accel component due to road roll
      - v_ego        (float): vehicle longitudinal velocity (m/s)
      - a_ego        (float): vehicle longitudinal acceleration (m/s^2)

future_plan : FuturePlan namedtuple
    Lookahead of up to 50 future values (5 seconds at 10 Hz) with fields:
      - lataccel      (list[float]): future target lateral accelerations
      - roll_lataccel  (list[float]): future road-roll lateral accelerations
      - v_ego          (list[float]): future vehicle velocities
      - a_ego          (list[float]): future vehicle longitudinal accelerations

Returns
-------
float
    Steering command. The simulator clips this to [-2, 2].

Cost function
-------------
Evaluated over steps 100-500:
    lataccel_cost = mean((target - predicted)^2) * 100
    jerk_cost     = mean((diff(predicted) / 0.1)^2) * 100
    total_cost    = lataccel_cost * 50 + jerk_cost

Lower is better. Tracking accuracy (lataccel_cost) is weighted 50x over
smoothness (jerk_cost), but jerk still matters.

Simulator constraints
---------------------
- Steering output clipped to [-2, 2]
- Lateral acceleration change clipped to +/-0.5 per step
- Autoregressive: controller actions affect future model predictions
"""

import numpy as np


class Controller:
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff
