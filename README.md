# Metallographic Polisher State Estimator (Kalman Filter Integration)

A hardware-agnostic Linear Kalman Filter that estimates the angle and angular
velocity (RPM) of a variable-speed metallographic polishing wheel from a single
noisy encoder angle. It runs on synthetic data today and is structured so that
moving to an ESP32 / Arduino / MicroPython board only requires changing the
data input and the noise tuning numbers.

# Result as shown in file: kalman_result.png

Metric (10 s run, 100 Hz, after 0.5 s start-up) ,  Speed RMSE 

Raw derived speed (dθ/dt) , 6.74 rad/s (~64 RPM) 
Kalman filtered speed , 1.50 rad/s (~14 RPM) 

# Overview

Metallographic polishing needs a stable, known wheel speed, especially at low RPM
where a sample pressed on the wheel can drag the speed down sharply. Cheap,
locally sourced encoders (or electrical noise near a motor or VFD) give a jittery
angle, and differentiating it amplifies the noise.

A low-pass filter can smooth that speed signal, but only by trading noise for
**lag**: the controller sees a load change late. A Kalman filter adds a physical
model (angle advances by speed × dt), so it:

tracks sudden load changes quickly, with minimal lag,
stays smooth at low RPM, where few encoder counts arrive per loop,
handles irregular loop timing, because `predict(dt)` takes the real `dt`.

# Filter Interface

Input:continuous (unwrapped) angle in radians and the elapsed time `dt`.
Output:filtered angle (rad), filtered speed (rad/s) and `.rpm`.
Consumer: a PID loop (or VFD speed reference) uses the filtered speed as feedback.

# Quick Start

```bash
pip install -r requirements.txt
python main.py
python main.py --save assets/kalman_result.png --no-show
```

The simulation runs 10 s at 100 Hz: the wheel starts at 50 rad/s (~478 RPM), a
sample is pressed at t = 3 s (speed drops to 15 rad/s) and lifted at t = 7 s.
Gaussian noise (σ = 0.05 rad) mimics a cheap encoder or EMI.

# Hardware Integration

`kalman_filter.py` never touches hardware. On a real machine, replace only the
data source in the loop in `main.py`.

1. Get a real angle. Count encoder ticks in an interrupt (or a hardware
counter such as ESP32 PCNT) and convert:

```
angle_rad = ticks * (2 * pi / counts_per_revolution)
```

Keep the angle unwrapped (never reset at 2π), and include x4 quadrature decoding
in `counts_per_revolution` if you use both channels.

2. Measure `dt` every loop. Never assume a fixed rate:

```cpp
unsigned long lastMicros = micros();

void loop() {
  unsigned long now = micros();
  float dt = (now - lastMicros) * 1e-6f;   // unsigned subtraction handles rollover
  lastMicros = now;
  if (dt <= 0.0f) return;

  kf.predict(dt);
  kf.update(readAngleRad());               // your encoder read
  float speed = kf.velocity;               // rad/s -> feed the PID
}
```

In MicroPython use `time.ticks_us()` and `time.ticks_diff()`.

3. Port the filter.The class is plain scalar maths (no NumPy), so it runs
unchanged on MicroPython and ports line by line to C++.

# Tuning

Two numbers in `main.py` (or your firmware):

```python
KF_R = ANGLE_NOISE_STD ** 2   # measurement noise variance (rad^2)
KF_Q = 300.0                  # process noise intensity (rad^2/s^3)
```

R (sensor jitter):measure it, do not guess. Log the angle with the wheel
still (or at constant speed, minus a straight-line fit) for 10 to 30 s, compute
the standard deviation σ in radians, and set `R = sigma ** 2`.

Q (load disturbance):how hard the motor or load can change the wheel speed.
Strong motor, light wheel or heavy samples need a larger Q. Fix R, start with
Q ≈ 100, press a sample on the wheel, and raise Q 2 to 3 times at a time until the
estimate follows the speed drop without visible lag. Use the lowest Q that d

