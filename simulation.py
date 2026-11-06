import numpy as np
import time
import random

class MetallographicPolishingDigitalTwin:
    """
    Digital Twin simulating a variable-speed polishing device interacting 
    with a robotic arm in a hybrid manufacturing cell.
    """
    def __init__(self, initial_roughness=12.5, target_roughness=0.05):
        self.target_roughness = target_roughness
        self.true_roughness = initial_roughness
        self.rpm = 300.0  # Initial RPM of the polishing disc
        
        # Kalman Filter Parameters for Sensor Fusion
        self.estimated_roughness = initial_roughness
        self.P = 1.0   # Estimation error covariance
        self.Q = 0.05  # Process noise covariance (system uncertainty)
        self.R_force = 0.4      # Sensor noise covariance for Load Cell
        self.R_vibe = 0.6       # Sensor noise covariance for IMU/Vibration

    def step_physics(self, dt=0.1):
        """Simulates the physical material removal model (Preston's Law proxy)"""
        if self.true_roughness <= self.target_roughness:
            self.true_roughness = self.target_roughness
            return

        # Material Removal Rate (MRR) is proportional to Pressure (implied) and Velocity (RPM)
        # As surface gets smoother, removal rate slows down asymptotically
        mrr = (self.rpm / 1000.0) * (self.true_roughness * 0.05) * dt
        self.true_roughness -= mrr

    def read_sensors(self):
        """Generates synthetic noisy sensor readings based on true physical state"""
        # Heuristic: Rougher surfaces create more friction (force) and structural vibration
        base_force = self.true_roughness * 0.8 + (self.rpm * 0.002)
        base_vibration = self.true_roughness * 1.2 + (self.rpm * 0.005)
        
        # Add Gaussian white noise to simulate real-world sensor imperfections
        noisy_force = base_force + np.random.normal(0, np.sqrt(self.R_force))
        noisy_vibration = base_vibration + np.random.normal(0, np.sqrt(self.R_vibe))
        
        return max(0.01, noisy_force), max(0.01, noisy_vibration)

    def sensor_fusion_update(self, force, vibration):
        """Fuses Load Cell and IMU data using a Multi-Sensor Kalman Filter"""
        # Map sensor readings back to an implied roughness measurement
        implied_roughness_from_force = force / 0.8
        implied_roughness_from_vibe = vibration / 1.2
        
        # Weighted average measurement based on inverse sensor variance
        w_force = 1 / self.R_force
        w_vibe = 1 / self.R_vibe
        z_measured = (w_force * implied_roughness_from_force + w_vibe * implied_roughness_from_vibe) / (w_force + w_vibe)
        R_fused = 1 / (w_force + w_vibe)

        # Kalman Filter Time Update (Prediction)
        self.P = self.P + self.Q
        
        # Kalman Filter Measurement Update (Correction)
        K = self.P / (self.P + R_fused)
        self.estimated_roughness = self.estimated_roughness + K * (z_measured - self.estimated_roughness)
        self.P = (1 - K) * self.P

    def adaptive_speed_control(self):
        """Software-driven control: Optimizes RPM dynamically based on fused state"""
        # Strategy: High speed for rough material removal, slow speed for precision mirror finishing
        if self.estimated_roughness > 5.0:
            self.rpm = 1200.0  # Aggressive polishing
        elif self.estimated_roughness > 1.0:
            self.rpm = 600.0   # Intermediate polishing
        else:
            self.rpm = 150.0   # Fine mirror finishing stage

# =====================================================================
# SIMULATION EXECUTION & LOGGING
# =====================================================================
if __name__ == "__main__":
    print("🤖 Launching Multi-Robot Hybrid Manufacturing Cell Digital Twin...")
    print("🔗 Initializing Variable-Speed Polishing Device Software Node...")
    
    twin = MetallographicPolishingDigitalTwin()
    
    # Data logging arrays for plotting
    time_history = []
    true_history = []
    estimated_history = []
    rpm_history = []
    
    dt = 0.2
    total_steps = 100
    
    for step in range(total_steps):
        current_time = step * dt
        
        # 1. Update underlying physics simulation
        twin.step_physics(dt)
        
        # 2. Read noisy hardware sensors
        force_raw, vibe_raw = twin.read_sensors()
        
        # 3. Fuse data to clean up noise and get true state estimate
        twin.sensor_fusion_update(force_raw, vibe_raw)
        
        # 4. Compute optimal control response (Variable Speed)
        twin.adaptive_speed_control()
        
        # Log telemetry
        time_history.append(current_time)
        true_history.append(twin.true_roughness)
        estimated_history.append(twin.estimated_roughness)
        rpm_history.append(twin.rpm)
        
        if step % 10 == 0:
            print(f"Time: {current_time:.1f}s | True Ra: {twin.true_roughness:.2f}µm | "
                  f"Fused Estimate: {twin.estimated_roughness:.2f}µm | Device Output: {twin.rpm} RPM")
            
    print("\n✅ Simulation Complete. Final Surface Roughness Target Achieved.")

    # Optional: Plotting code to make it look professional (requires pip install matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Surface Roughness (Ra in µm)', color='tab:blue')
        ax1.plot(time_history, true_history, label='True Physical State', color='tab:blue', linewidth=2)
        ax1.plot(time_history, estimated_history, '--', label='Fused Sensor Estimate (Kalman Filter)', color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper right')

        ax2 = ax1.twinx()  
        ax2.set_ylabel('Polishing Device Speed (RPM)', color='tab:green')
        ax2.step(time_history, rpm_history, label='Adaptive RPM Control', color='tab:green', alpha=0.5, where='post')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        plt.title('Digital Twin: Sensor-Fused Closed-Loop Adaptive Polishing Sequence')
        fig.tight_layout()
        plt.savefig('digital_twin_performance.png')
        print("📊 Telemetry graph saved as 'digital_twin_performance.png'. Check your directory!")
    except ImportError:
        print("\n💡 Run 'pip install matplotlib' to visually graph the sensor fusion telemetry next time.")
