"""
kalman_filter.py
================

A tiny, dependency-free Linear Kalman Filter for estimating the angle and
angular velocity of a rotating polishing wheel from a single noisy angle
measurement (e.g. an incremental encoder).

Design goals
------------
* Hardware-agnostic: the filter knows nothing about encoders, motors,
  plotting or simulation. It only receives ``dt`` and a measured angle.
* Portable: all 2x2 matrix algebra is written out as plain scalar maths,
  so this file ports line-by-line to C++ (Arduino / ESP32) or runs as-is
  on MicroPython, where NumPy is not available.
* Variable time step: ``predict(dt)`` takes ``dt`` on every call, so
  irregular loop timing is handled correctly.

State vector
------------
    x = [ angle (rad), angular velocity (rad/s) ]^T

Measurement
-----------
    z = measured angle (rad)   -- continuous / "unwrapped", NOT limited to 0..2*pi
"""

import math


class PolisherKalmanFilter:
    """Two-state (angle, angular velocity) Linear Kalman Filter.

    The matrices of the filter, in plain language
    ---------------------------------------------
    F  (state transition, 2x2)
        "How the wheel moves by itself in ``dt`` seconds."
        ``F = [[1, dt], [0, 1]]`` means: new angle = old angle + speed * dt,
        and the speed stays the same (constant-velocity model). Real speed
        changes (a sample being pressed on the wheel) are not modelled
        explicitly -- they are absorbed by Q (see below).

    H  (measurement matrix, 1x2)
        "What the sensor can actually see."
        ``H = [1, 0]`` means the encoder measures the angle only. Speed is
        never measured directly; the filter infers it.

    Q  (process noise covariance, 2x2)
        "How much we distrust the constant-velocity model."
        It represents unmodelled torque disturbances (load changes, belt
        slip, motor ripple). It is built from a single tuning number ``q``
        (units: (rad/s^2)^2 * s). A LARGER ``q`` lets the estimate react
        faster to sudden load changes but passes through more sensor noise.
        ``Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]``

    R  (measurement noise covariance, 1x1)
        "How much we distrust the sensor."
        Variance of the angle measurement noise in rad^2
        (``R = sigma_angle^2``). A LARGER ``R`` makes the filter smoother
        but slower to follow real changes.

    P  (estimate covariance, 2x2)
        "How unsure the filter is about its own estimate right now."
        Diagonal terms are the variances of angle and speed; the off-diagonal
        term says how the two errors are correlated. ``P`` grows during
        ``predict`` (time passes, uncertainty grows) and shrinks during
        ``update`` (a measurement arrives). It is stored as three numbers
        (``p00``, ``p01``, ``p11``) because it is symmetric.

    K  (Kalman gain, 2x1)
        Computed internally each update. It decides how much of the
        measurement error ("innovation") is used to correct the angle and
        the speed.
    """

    def __init__(self,
                 process_noise_q,
                 measurement_noise_r,
                 initial_angle=0.0,
                 initial_velocity=0.0,
                 initial_velocity_std=50.0):
        """Create the filter.

        Args:
            process_noise_q: Tuning scalar ``q`` for Q (rad^2/s^3). Must be > 0.
            measurement_noise_r: Angle measurement variance R in rad^2. Must be > 0.
            initial_angle: Starting angle guess in rad (use first reading).
            initial_velocity: Starting speed guess in rad/s.
            initial_velocity_std: How wrong the starting speed guess may be
                (rad/s). Large values make the filter converge quickly.
        """
        if process_noise_q <= 0.0 or measurement_noise_r <= 0.0:
            raise ValueError("process_noise_q and measurement_noise_r must be > 0")

        self.q = float(process_noise_q)
        self.r = float(measurement_noise_r)

        # State estimate.
        self.angle = float(initial_angle)
        self.velocity = float(initial_velocity)

        # Covariance P = [[p00, p01], [p01, p11]].
        self.p00 = self.r                       # trust the first reading to within R
        self.p01 = 0.0
        self.p11 = float(initial_velocity_std) ** 2

        # Diagnostics from the most recent update (useful for load detection).
        self.innovation = 0.0                   # measured - predicted angle (rad)
        self.innovation_variance = self.r       # S = H P H^T + R (rad^2)

    # ------------------------------------------------------------------ #
    # Core filter steps
    # ------------------------------------------------------------------ #
    def predict(self, dt):
        """Propagate the state and covariance forward by ``dt`` seconds.

        Computes ``x = F x`` and ``P = F P F^T + Q``.

        Args:
            dt: Time since the previous call, in seconds. Must be > 0.
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        # x = F x  (angle advances by velocity * dt; velocity unchanged)
        self.angle += self.velocity * dt

        # P = F P F^T + Q, expanded by hand for the 2x2 case.
        p00, p01, p11 = self.p00, self.p01, self.p11
        dt2 = dt * dt
        dt3 = dt2 * dt

        self.p00 = p00 + 2.0 * dt * p01 + dt2 * p11 + self.q * dt3 / 3.0
        self.p01 = p01 + dt * p11 + self.q * dt2 / 2.0
        self.p11 = p11 + self.q * dt

    def update(self, measured_angle):
        """Correct the prediction with a new angle measurement.

        Args:
            measured_angle: Encoder angle in rad (continuous, unwrapped).

        Returns:
            Tuple ``(angle, velocity)`` of the corrected estimate.
        """
        # Innovation y = z - H x  and its variance S = H P H^T + R.
        y = measured_angle - self.angle
        s = self.p00 + self.r

        # Kalman gain K = P H^T / S  (H = [1, 0] picks the first column of P).
        k0 = self.p00 / s
        k1 = self.p01 / s

        # State correction x = x + K y.
        self.angle += k0 * y
        self.velocity += k1 * y

        # Covariance correction P = (I - K H) P, expanded by hand.
        # This form keeps P exactly symmetric.
        self.p11 = self.p11 - k1 * self.p01
        self.p01 = (1.0 - k0) * self.p01
        self.p00 = (1.0 - k0) * self.p00

        self.innovation = y
        self.innovation_variance = s
        return self.angle, self.velocity

    def step(self, dt, measured_angle):
        """Convenience wrapper: ``predict(dt)`` followed by ``update(...)``."""
        self.predict(dt)
        return self.update(measured_angle)

    # ------------------------------------------------------------------ #
    # Convenience read-outs
    # ------------------------------------------------------------------ #
    @property
    def rpm(self):
        """Estimated wheel speed in revolutions per minute."""
        return self.velocity * 60.0 / (2.0 * math.pi)

    @property
    def normalized_innovation_squared(self):
        """Innovation^2 / S. Values far above ~9 flag a surprise (e.g. a sudden
        load change) because the angle moved more than the model expected."""
        return (self.innovation ** 2) / self.innovation_variance
