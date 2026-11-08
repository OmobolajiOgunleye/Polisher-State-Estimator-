"""
main.py
=======

Simulation and visualisation for the Variable-Speed Metallographic Polishing
Device State Estimator.

Scenario (10 s at 100 Hz):
    * 0.0 s  : wheel spinning at 50 rad/s (~478 RPM)
    * 3.0 s  : heavy sample pressed on the wheel -> speed drags down to 15 rad/s
    * 7.0 s  : sample lifted -> motor recovers to 50 rad/s

The "encoder" returns the true angle plus heavy Gaussian noise (cheap sensor /
EMI). The Kalman filter in ``kalman_filter.py`` estimates the speed from those
noisy angles only.

Usage:
    python main.py                         # show interactive window
    python main.py --save assets/result.png --no-show
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from kalman_filter import PolisherKalmanFilter

# --------------------------------------------------------------------------- #
# Simulation parameters
# --------------------------------------------------------------------------- #
DT = 0.01                    # s, 100 Hz loop
DURATION = 10.0              # s
T_LOAD_ON = 3.0              # s, sample pressed against the wheel
T_LOAD_OFF = 7.0             # s, sample lifted
SPEED_FREE = 50.0            # rad/s (~478 RPM)
SPEED_LOADED = 15.0          # rad/s (~143 RPM)
TAU_LOAD = 0.15              # s, how fast the load drags the wheel down
TAU_RECOVER = 0.40           # s, how fast the motor recovers after lift-off

ANGLE_NOISE_STD = 0.05       # rad, "cheap encoder / EMI" noise (~2.9 degrees)

# Kalman tuning (see README "Tuning Instructions")
KF_R = ANGLE_NOISE_STD ** 2  # rad^2, measurement noise variance
KF_Q = 300.0                 # rad^2/s^3, process noise intensity (load agility)

RANDOM_SEED = 42


def rad_s_to_rpm(w):
    """Convert rad/s to RPM."""
    return w * 60.0 / (2.0 * math.pi)


def rpm_to_rad_s(rpm):
    """Convert RPM to rad/s."""
    return rpm * 2.0 * math.pi / 60.0


def simulate_true_wheel(t):
    """Integrate a simple first-order wheel model through the load scenario.

    Args:
        t: Array of sample times in seconds.

    Returns:
        Tuple ``(true_angle, true_speed)`` arrays (rad, rad/s).
    """
    n = len(t)
    angle = np.zeros(n)
    speed = np.zeros(n)
    speed[0] = SPEED_FREE

    for k in range(1, n):
        dt = t[k] - t[k - 1]
        time_now = t[k]

        if T_LOAD_ON <= time_now < T_LOAD_OFF:
            target, tau = SPEED_LOADED, TAU_LOAD
        else:
            target, tau = SPEED_FREE, TAU_RECOVER

        # First-order response toward the target speed, then integrate angle.
        speed[k] = speed[k - 1] + (target - speed[k - 1]) * dt / tau
        angle[k] = angle[k - 1] + speed[k - 1] * dt

    return angle, speed


def run_simulation():
    """Generate data, run the filter, and return everything needed to plot."""
    rng = np.random.default_rng(RANDOM_SEED)

    t = np.arange(0.0, DURATION + DT / 2, DT)
    true_angle, true_speed = simulate_true_wheel(t)

    # --- "Sensor": true angle plus heavy Gaussian noise ------------------- #
    noisy_angle = true_angle + rng.normal(0.0, ANGLE_NOISE_STD, size=t.shape)

    # --- Naive speed: finite difference of the noisy angle ---------------- #
    raw_speed = np.full_like(t, np.nan)
    raw_speed[1:] = np.diff(noisy_angle) / np.diff(t)

    # --- Kalman filter loop (this is the part that runs on the MCU) ------- #
    kf = PolisherKalmanFilter(process_noise_q=KF_Q,
                              measurement_noise_r=KF_R,
                              initial_angle=noisy_angle[0],
                              initial_velocity=0.0,        # deliberately unknown
                              initial_velocity_std=60.0)

    kf_speed = np.zeros_like(t)
    kf_angle = np.zeros_like(t)
    kf_speed[0], kf_angle[0] = kf.velocity, kf.angle

    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]                 # on hardware: measured loop time
        kf.predict(dt)
        kf.update(noisy_angle[k])
        kf_angle[k], kf_speed[k] = kf.angle, kf.velocity

    return {
        "t": t,
        "true_angle": true_angle,
        "true_speed": true_speed,
        "noisy_angle": noisy_angle,
        "raw_speed": raw_speed,
        "kf_speed": kf_speed,
        "kf_angle": kf_angle,
    }


def rmse(estimate, truth, mask):
    """Root-mean-square error over the samples selected by ``mask``."""
    return float(np.sqrt(np.nanmean((estimate[mask] - truth[mask]) ** 2)))


def plot_results(d, save_path=None, show=True):
    """Create the three-line comparison plot."""
    t = d["t"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.axvspan(T_LOAD_ON, T_LOAD_OFF, color="#f4a261", alpha=0.15,
               label="Sample pressed on wheel")

    ax.plot(t, d["raw_speed"], color="#b0b0b0", linewidth=0.9, alpha=0.9,
            label="Raw noisy derived speed (d\u03b8/dt)", zorder=1)
    ax.plot(t, d["true_speed"], color="#1d3557", linewidth=2.4,
            linestyle="--", label="True speed", zorder=3)
    ax.plot(t, d["kf_speed"], color="#e63946", linewidth=2.2,
            label="Kalman filtered speed", zorder=2)

    ax.set_title("Metallographic Polisher State Estimator \u2013 Kalman Filter "
                 "vs. Raw Derived Speed", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_xlim(0.0, DURATION)
    ax.set_ylim(-5.0, 100.0)
    ax.legend(loc="upper right", frameon=True)

    # Secondary axis in RPM for shop-floor intuition.
    ax_rpm = ax.secondary_yaxis("right", functions=(rad_s_to_rpm, rpm_to_rad_s))
    ax_rpm.set_ylabel("Speed (RPM)")

    ax.annotate("Load applied (3 s)", xy=(T_LOAD_ON, 45), xytext=(1.0, 75),
                arrowprops=dict(arrowstyle="->", color="#444"), color="#444")
    ax.annotate("Sample lifted (7 s)", xy=(T_LOAD_OFF, 20), xytext=(7.4, 5),
                arrowprops=dict(arrowstyle="->", color="#444"), color="#444")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--save", metavar="PATH", help="save the figure to PATH")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open an interactive window")
    args = parser.parse_args()

    data = run_simulation()

    # Ignore the first 0.5 s so the filter's start-up convergence does not
    # dominate the error figures.
    mask = data["t"] >= 0.5
    print("Speed RMSE after 0.5 s (rad/s):")
    print(f"  Raw derived : {rmse(data['raw_speed'], data['true_speed'], mask):6.2f}")
    print(f"  Kalman      : {rmse(data['kf_speed'], data['true_speed'], mask):6.2f}")

    plot_results(data, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
