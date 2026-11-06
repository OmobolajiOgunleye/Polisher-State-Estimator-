import numpy as np

class MetallographicPolishingDigitalTwin:
    """
    State-space Digital Twin simulating an adaptive polishing device 
    operating within an automated hybrid manufacturing cell.
    """
    def __init__(self, initial_roughness=12.5, target_roughness=0.05):
        self.target_roughness = target_roughness
        self.true_roughness = initial_roughness
        self.rpm = 1200.0  # Core working velocity
        
        # State Estimation Parameters (Kalman Filter Configuration)
        self.estimated_roughness = initial_roughness
        self.P = 1.0       # Estimation error covariance matrix initialization
        self.Q = 0.01      # Process variance (system dynamic uncertainty)
        self.R_force = 0.4 # Measurement noise covariance: Force Transducer
        self.R_vibe = 0.6  # Measurement noise covariance: Accelerometer Array

    def step_physics(self, dt=0.2):
        """Applies empirical material removal rate equations (Preston's Law)"""
        if self.true_roughness <= self.target_roughness:
            self.true_roughness = self.target_roughness
            return

        # Material Removal Rate (MRR) modeled as a function of velocity (RPM) and state metrics
        mrr = (self.rpm / 1200.0) * (self.true_roughness * 0.04) * dt
        self.true_roughness -= mrr

    def read_sensors(self):
        """Simulates data acquisition streams with characteristic environmental noise"""
        # Linear correlation: Structural vibration and shear force scale with sample profile peaks
        base_force = (self.true_roughness * 0.8) + (self.rpm * 0.002)
        base_vibration = (self.true_roughness * 1.2) + (self.rpm * 0.005)
        
        # Superimpose Gaussian white noise components
        noisy_force = base_force + np.random.normal(0, np.sqrt(self.R_force))
        noisy_vibration = base_vibration + np.random.normal(0, np.sqrt(self.R_vibe))
        
        return max(0.01, noisy_force), max(0.01, noisy_vibration)

    def sensor_fusion_update(self, force, vibration):
        """Executes sensor data convergence using a multi-channel linear estimator"""
        # Mathematical Inversion: Isolate estimated parameter metrics from composite sensor data
        implied_roughness_from_force = (force - (self.rpm * 0.002)) / 0.8
        implied_roughness_from_vibe = (vibration - (self.rpm * 0.005)) / 1.2
        
        # Compute measurement variance optimization weight coefficients
        w_force = 1.0 / self.R_force
        w_vibe = 1.0 / self.R_vibe
        
        z_measured = (w_force * implied_roughness_from_force + w_vibe * implied_roughness_from_vibe) / (w_force + w_vibe)
        R_fused = 1.0 / (w_force + w_vibe)

        # Time-update prediction step
        self.P = self.P + self.Q
        
        # Measurement-update correction step (Kalman Gain calculations)
        K = self.P / (self.P + R_fused)
        self.estimated_roughness = self.estimated_roughness + K * (z_measured - self.estimated_roughness)
        self.P = (1.0 - K) * self.P

    def adaptive_speed_control(self):
        """Dynamic operational speed scaling based on decoupled state estimations"""
        if self.estimated_roughness > 8.0:
            self.rpm = 1200.0  # Bulk material removal mode
        elif self.estimated_roughness > 4.0:
            self.rpm = 600.0   # Planarization rectification mode
        else:
            self.rpm = 150.0   # Ultra-precision finishing mode

if __name__ == "__main__":
    print("[SYS_INFO] Initializing multi-robot node sub-system telemetry simulation...")
    
    twin = MetallographicPolishingDigitalTwin()
    
    time_history, true_history, estimated_history, rpm_history = [], [], [], []
    dt = 0.2
    total_steps = 100
    
    for step in range(total_steps):
        current_time = step * dt
        
        twin.step_physics(dt)
        force_raw, vibe_raw = twin.read_sensors()
        twin.sensor_fusion_update(force_raw, vibe_raw)
        twin.adaptive_speed_control()
        
        time_history.append(current_time)
        true_history.append(twin.true_roughness)
        estimated_history.append(twin.estimated_roughness)
        rpm_history.append(twin.rpm)
        
        if step % 10 == 0:
            print(f"t: {current_time:.1f}s | True_Ra: {twin.true_roughness:.3f} um | "
                  f"Est_Ra: {twin.estimated_roughness:.3f} um | Control_Out: {twin.rpm:.1f} RPM")
            
    print("[SYS_INFO] Iteration sequence terminated nominal status.")

    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Surface Roughness (Ra in um)', color='tab:blue')
        ax1.plot(time_history, true_history, label='True State Model', color='tab:blue', linewidth=2)
        ax1.plot(time_history, estimated_history, '--', label='Fused Observation Vector (Kalman Filter)', color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper right')

        ax2 = ax1.twinx()  
        ax2.set_ylabel('Actuator Speed Target (RPM)', color='tab:green')
        ax2.step(time_history, rpm_history, label='Dynamic Parameter Scaling', color='tab:green', alpha=0.6, where='post')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        plt.title('Polishing Node Closed-Loop System State Profiles')
        fig.tight_layout()
        plt.savefig('digital_twin_performance.png')
    except ImportError:
        pass
