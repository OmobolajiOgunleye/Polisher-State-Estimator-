Cyber-Physical Digital Twin for Autonomous Surface Finishing in Multi-Robot Hybrid Manufacturing

Project Overview
This repository contains the software framework and digital twin architecture designed to elevate a variable-speed metallographic polishing device into an intelligent, networked asset for multi-robot hybrid manufacturing cells. 

This project shifts the traditional hardware-centric paradigm of sample preparation into an autonomous, closed-loop cyber-physical system (CPS). By leveraging synthetic telemetry streams, this framework validates multi-sensor fusion algorithms and adaptive real-time control strategies required for automated surface finishing.

Key Software Features
i) Multi-Sensor Sensor Fusion: Implements a state-space Multi-Sensor Kalman Filter that fuses high-frequency structural vibration data (IMU proxy) and friction/normal force data (Load Cell proxy) to filter out industrial environmental noise.
ii) Adaptive Control Loops: Replaces manual control with an intelligent software agent that dynamically modifies device RPM based on the estimated real-time surface roughness profile.
iii) Deterministic Physics Engine:Simulates material removal rates (MRR) based on a modified Preston’s Law framework.

System Architecture
The Python control node is split into four distinct layers to ensure modular deployment into ROS 2 or industrial edge computing devices:
i) Physics Simulation Layer: Handles the continuous-time degradation of material surface peaks.
ii) Sensor Modality Layer: Simulates real-world sensor imperfections by injecting Gaussian white noise into telemetry streams.
iii) Estimation Layer (Kalman Filter):Tracks real-time surface transition states and calculates estimation error covariance.
iv) Decision & Actuation Layer:Executes a step-down RPM profiling strategy to transition seamlessly from aggressive material removal to a sub-micron mirror finish.

Installation & Execution

1. Clone the repository:
   bash
   git clone https://github.com
   cd hybrid-mfg-polishing-digital-twin
   

2. Install dependencies:
   bash
   pip install numpy matplotlib
   

3. Run the core simulation node:
   bash
   python simulation.py
   

Expected Output
Upon execution, the script generates real-time terminal telemetry logs and exports a performance profile graph (digital_twin_performance.png). The graph maps how the fused sensor estimate accurately tracks the true physical state of the material while the adaptive controller automatically steps down device RPM as the target finish is reached.
