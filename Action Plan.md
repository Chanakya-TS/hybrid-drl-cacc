# **LLM ACTION PLAN: Implementation and Results for Hybrid DRL-MPC Framework**

## **OVERALL GOAL**

To generate the Python code and simulation data required to validate the hybrid DRL-MPC eco-driving controller. This involves setting up a simulation, implementing all controllers, running experiments, and processing the data into the tables and plots described in the paper plan.

## **PERSONA**

Adopt the persona of a senior robotics engineer specializing in autonomous vehicle simulation and control. Your code should be well-structured, commented, and follow Python best practices (PEP 8). Use numpy for all numerical operations.

### **PHASE 1: SIMULATION ENVIRONMENT SETUP**

**STEP 1: DEFINE PROJECT STRUCTURE AND DEPENDENCIES**

* **Objective:** Establish a clean project structure and list the required Python libraries.  
* **Action:**  
  1. Create a root directory for the project.  
  2. Create subdirectories: /controllers, /environment, /utils, /results.  
  3. List the necessary Python libraries in a requirements.txt file:
     numpy
     matplotlib
     pandas
     scipy
     stable-baselines3\[extra\]
     gymnasium
     casadi
     carla==0.9.15
     pygame

  4. **CARLA Setup:**
     * Download CARLA from https://github.com/carla-simulator/carla/releases (version 0.9.13+)
     * Extract and ensure CarlaUE4.exe is accessible
     * Install CARLA Python API matching your CARLA server version
     * The simulation will use CARLA's high-fidelity vehicle physics for realistic dynamics

**STEP 2: IMPLEMENT CARLA ENVIRONMENT WRAPPER**

* **Objective:** Create a CARLA-based car-following environment with realistic vehicle dynamics.
* **Action:** In /environment/car\_following.py, create a CarFollowingEnv class.
  1. **\_\_init\_\_(self, carla\_host='localhost', carla\_port=2000, lead\_vehicle\_trajectory=None):**
     * Connect to CARLA server: `self.client = carla.Client(host, port)`
     * Set timeout and get world: `self.world = self.client.get_world()`
     * **Enable synchronous mode** with fixed time step (dt = 0.05 or 0.1s):
       ```python
       settings = self.world.get_settings()
       settings.synchronous_mode = True
       settings.fixed_delta_seconds = 0.05
       self.world.apply_settings(settings)
       ```
     * Store lead vehicle trajectory data for controlled scenario
     * Initialize variables: self.ego\_vehicle = None, self.lead\_vehicle = None
     * Set simulation timestep dt

  2. **reset(self):**
     * Destroy any existing vehicles from previous episodes
     * **Spawn ego vehicle:**
       - Get vehicle blueprint: `bp = blueprint_library.find('vehicle.tesla.model3')`
       - Choose spawn point on a straight road (use `world.get_map().get_spawn_points()`)
       - Spawn: `self.ego_vehicle = world.spawn_actor(bp, spawn_point)`
     * **Spawn lead vehicle:**
       - Spawn ahead of ego vehicle (e.g., 30m forward)
       - Either use Traffic Manager with set\_target\_velocity() or custom trajectory control
     * Initialize data collection lists
     * Return initial state: \[ego\_velocity, relative\_velocity, distance\_gap\]

  3. **step(self, acceleration):**
     * **Convert acceleration command to CARLA control:**
       - If acceleration > 0: apply throttle (use PID or simple mapping)
       - If acceleration < 0: apply brake (negative acceleration → brake value)
       - Create: `control = carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0)`
     * Apply control: `self.ego_vehicle.apply_control(control)`
     * **Tick simulation:** `self.world.tick()` (synchronous mode)
     * **Update lead vehicle:** Apply next trajectory point or Traffic Manager handles it
     * **Extract state from CARLA:**
       - ego\_velocity = get\_velocity() (convert to scalar: sqrt(vx²+vy²+vz²))
       - ego\_position = get\_location()
       - lead\_velocity = lead\_vehicle.get\_velocity()
       - lead\_position = lead\_vehicle.get\_location()
       - distance\_gap = distance between positions
       - relative\_velocity = lead\_velocity - ego\_velocity
     * **Calculate energy consumption:**
       - current\_throttle = control.throttle
       - energy\_increment = throttle * velocity * dt
       - Store for logging
     * Return state observation: \[ego\_velocity, relative\_velocity, distance\_gap\]

  4. **destroy(self):**
     * Clean up: destroy ego and lead vehicles
     * Restore asynchronous mode if needed
     * Close connection

  5. **Helper Methods:**
     * `_acceleration_to_control(acceleration, current_velocity)`: Maps desired acceleration to throttle/brake
     * `_get_vehicle_speed(vehicle)`: Extract scalar speed from velocity vector
     * `_get_distance_between_vehicles()`: Calculate longitudinal distance gap

### **PHASE 2: CONTROLLER IMPLEMENTATION**

**STEP 3: IMPLEMENT THE LOW-LEVEL MPC CONTROLLER**

* **Objective:** Create a reusable MPC class that outputs desired acceleration.
* **Action:** In /controllers/mpc\_controller.py, create an MPCController class.
  1. **\_\_init\_\_(self, N, dt):**
     * N: prediction horizon (e.g., 20-50 steps)
     * dt: timestep matching CARLA simulation (0.05 or 0.1s)
     * Initialize CasADi optimization problem

  2. **Setup Optimization Problem:**
     * Use `casadi.Opti()` to define optimization
     * **Decision variables:**
       - States X: position and velocity over horizon N
       - Controls U: acceleration over horizon N
     * **Note:** MPC uses simplified kinematic model for prediction, even though CARLA uses full dynamics

  3. **Define Cost Function:**
     * Create parameterized cost function with weights \[w\_v, w\_s, w\_u\]:
       ```
       J = w_v * Σ(v_ref - v[k])² +
           w_s * Σ(safety_slack[k])² +
           w_u * Σ(u[k]²)
       ```
     * Velocity tracking term: penalizes deviation from desired/lead velocity
     * Safety term: penalizes safety constraint violations
     * Control effort term: penalizes aggressive acceleration (smoothness)

  4. **Define Constraints:**
     * **Dynamics constraints** (simplified kinematic model for prediction):
       - v\[k+1\] = v\[k\] + u\[k\] * dt
       - p\[k+1\] = p\[k\] + v\[k\] * dt + 0.5 * u\[k\] * dt²
     * **State limits:**
       - 0 <= v\[k\] <= v\_max (e.g., 30 m/s)
     * **Control limits:**
       - u\_min <= u\[k\] <= u\_max (e.g., -3 m/s² to 2 m/s²)
     * **Safety constraint:**
       - d\[k\] >= THW * v\_ego\[k\] where THW = 1.5-2.0 seconds

  5. **solve(self, initial\_state, lead\_trajectory, weights):**
     * Update cost function with new weights \[w\_v, w\_s, w\_u\] from DRL agent
     * Set initial condition: X\[:, 0\] == initial\_state
     * Update reference trajectory based on lead vehicle prediction
     * Solve optimization: `sol = opti.solve()`
     * **Return:** First optimal control action U\[0\] (desired acceleration)
     * **Note:** This acceleration will be mapped to CARLA throttle/brake by environment

**STEP 4: IMPLEMENT BASELINE CONTROLLERS**

* **Objective:** Create the controllers for comparison.  
* **Action:** In /controllers/baseline\_controllers.py:  
  1. **FixedWeightMPC class:** Inherits from MPCController. In its solve method, it calls the parent solve method with a fixed, hand-tuned set of weights.  
  2. **ACC\_Controller class:** Implement a simple Adaptive Cruise Control using a PID controller on the distance gap error. The setpoint for the distance gap should be based on a constant time headway policy (d\_setpoint \= THW \* ego\_velocity).

### **PHASE 3: DRL INTEGRATION AND TRAINING**

**STEP 5: CREATE A GYMNASIUM WRAPPER**

* **Objective:** Make the simulation compatible with stable-baselines3.  
* **Action:** In /environment/gym\_wrapper.py, create a DrlMpcEnv class that inherits from gym.Env.  
  1. **\_\_init\_\_(self, config):** Initialize the CarFollowingEnv and the MPCController.  
  2. **Define Spaces:**  
     * self.observation\_space: A gym.spaces.Box with normalized values for \[ego\_velocity, relative\_velocity, distance\_gap\].  
     * self.action\_space: A gym.spaces.Box for the MPC weights \[w\_v, w\_s, w\_u\]. The actions should be bounded (e.g., between 0 and 1).  
  3. **reset(self, seed=None):** Reset the underlying CarFollowingEnv to its initial state. Return the initial observation and info.  
  4. **step(self, action):**  
     * **This is the core loop:**  
     * Normalize the action (weights) so they sum to 1\.  
     * Call self.mpc\_controller.solve() with the current state and the DRL's chosen weights. This returns the optimal acceleration.  
     * Pass this acceleration to the self.car\_following\_env.step() method to update the simulation.  
     * Calculate the reward based on the new state (using the reward function from the paper plan).  
     * Check for terminal conditions (collision, end of episode).  
     * Return observation, reward, terminated, truncated, info.

**STEP 6: CREATE THE TRAINING SCRIPT**

* **Objective:** Train the DRL agent using CARLA simulation.
* **Prerequisites:** CARLA server must be running before starting training
  ```bash
  # Start CARLA server without rendering for faster training
  CarlaUE4.exe -RenderOffScreen -quality-level=Low -fps=20
  ```
* **Action:** In a train.py script:
  1. **Setup:**
     * Import required libraries
     * Set random seeds for reproducibility (numpy, random, CARLA if possible)
     * Define training hyperparameters (total\_timesteps, learning\_rate, etc.)

  2. **Create Environment:**
     * Instantiate DrlMpcEnv with CARLA connection settings
     * Optionally wrap with stable-baselines3 wrappers (Monitor, VecNormalize)

  3. **Initialize DRL Agent:**
     * Use Soft Actor-Critic (SAC) - good for continuous control
     * Configure policy network: `model = SAC("MlpPolicy", env, verbose=1)`
     * Consider hyperparameters:
       - learning\_rate: 3e-4
       - buffer\_size: 100000
       - batch\_size: 256
       - tau: 0.005

  4. **Training Loop:**
     * Call `model.learn(total_timesteps=50000-200000)`
     * Use callbacks for periodic model saving and logging
     * Monitor training progress (episode rewards, episode lengths)

  5. **Save Trained Model:**
     * `model.save("drl_mpc_agent")`
     * Save normalization statistics if using VecNormalize

  6. **Error Handling:**
     * Wrap in try-except to handle CARLA disconnections
     * Clean up CARLA actors on training interruption

### **PHASE 4 & 5: EXPERIMENTATION AND RESULTS**

**STEP 7: CREATE THE EVALUATION SCRIPT**

* **Objective:** Run all controllers on a standardized test scenario and collect data.  
* **Action:** In an evaluate.py script:  
  1. **Define a Test Scenario:** Create a challenging lead vehicle velocity profile (e.g., includes acceleration, cruising, hard braking, and stop-and-go).  
  2. **Load Controllers:**  
     * Load the trained DRL agent: drl\_model \= SAC.load("drl\_mpc\_agent").  
     * Instantiate the FixedWeightMPC.  
     * Instantiate the ACC\_Controller.  
  3. **Run Simulation Loop:** For each controller, run a full simulation episode using the test scenario.  
  4. **Log Data:** During each run, log all relevant time-series data (time, ego velocity, lead velocity, acceleration, distance gap, DRL-chosen weights) to a separate CSV file for each controller (e.g., results/hybrid\_run.csv, results/mpc\_run.csv).

**STEP 8: GENERATE RESULTS SCRIPT**

* **Objective:** Process the logged CSV data into final tables and plots.
* **Action:** In a generate\_results.py script:
  1. **Load Data:**
     * Use pandas to load CSV files from each controller run
     * Ensure all runs have matching time vectors for comparison

  2. **Calculate Quantitative Metrics:**
     For each controller's dataframe, calculate:

     **a. Total Energy Consumption:**
     * Primary method: `energy = Σ(throttle * velocity * dt)`
     * Alternative: `energy = m * Σ(max(acceleration, 0) * velocity * dt)` where m = vehicle mass
     * Units: Energy proxy (can normalize by trip distance)

     **b. RMS Jerk (Comfort Metric):**
     * Calculate jerk: `jerk[t] = (acceleration[t] - acceleration[t-1]) / dt`
     * RMS Jerk: `rms_jerk = sqrt(mean(jerk²))`
     * Units: m/s³

     **c. Minimum Time Headway (Safety Metric):**
     * Time headway: `THW[t] = distance_gap[t] / ego_velocity[t]`
     * Minimum THW: `min_thw = min(THW)` (excluding zero velocity)
     * Units: seconds

     **d. Average Speed (Efficiency):**
     * `avg_speed = mean(ego_velocity)`
     * Higher is generally better for efficiency

     **e. Control Smoothness:**
     * `throttle_std = std(diff(throttle))`
     * Lower indicates smoother control

  3. **Generate Markdown Table:**
     * Create formatted table with columns: \[Controller, Energy (J), RMS Jerk (m/s³), Min THW (s), Avg Speed (m/s)\]
     * Include percentage improvements of hybrid over baselines
     * Example:
       ```
       | Controller        | Energy  | RMS Jerk | Min THW | Avg Speed |
       |-------------------|---------|----------|---------|-----------|
       | Hybrid DRL-MPC    | 1234.5  | 0.45     | 1.8     | 18.2      |
       | Fixed-Weight MPC  | 1389.2  | 0.52     | 1.7     | 17.8      |
       | ACC               | 1456.8  | 0.61     | 2.1     | 17.5      |
       ```

  4. **Generate Plots:**
     **Figure 1 (Performance Comparison):**
     * Create 3-panel subplot (stacked vertically)
     * Panel 1: Velocity over time (ego and lead for all controllers)
     * Panel 2: Acceleration over time (all controllers overlaid)
     * Panel 3: Distance gap over time (all controllers, with THW safety line)
     * Use distinct colors/linestyles for each controller
     * Add legends, axis labels, grid

     **Figure 2 (Adaptive Weights):**
     * Time series plot of DRL-chosen weights \[w\_v, w\_s, w\_u\]
     * Show how weights adapt to different driving phases
     * Annotate key events (braking, acceleration, cruise)

     **Figure 3 (Optional - Energy Analysis):**
     * Bar chart comparing total energy consumption
     * Stacked bars showing energy breakdown if available

  5. **Save Results:**
     * Save plots as high-resolution PNG (300 dpi): `plt.savefig('results/performance_comparison.png', dpi=300)`
     * Save markdown table to file: `results/metrics_table.md`
     * Print summary statistics to console

---

## **CARLA-SPECIFIC CONSIDERATIONS**

### **Server Management**
* **Starting CARLA:**
  - Windows: `CarlaUE4.exe -quality-level=Low -fps=20`
  - For training (no rendering): `CarlaUE4.exe -RenderOffScreen -quality-level=Low`
  - Default port: 2000

* **Resource Management:**
  - CARLA requires significant GPU memory (4GB+ recommended)
  - Monitor memory usage during long training runs
  - Consider periodic server restarts for multi-hour training

### **Synchronous Mode (CRITICAL)**
Always use synchronous mode for deterministic, reproducible results:
```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 Hz
world.apply_settings(settings)
```

### **Actor Cleanup**
Prevent memory leaks by properly destroying actors:
```python
def cleanup(self):
    if self.ego_vehicle:
        self.ego_vehicle.destroy()
    if self.lead_vehicle:
        self.lead_vehicle.destroy()
```

### **Control Mapping**
**Throttle/Brake Conversion:**
* Never apply throttle and brake simultaneously
* Simple mapping example:
  ```python
  if acceleration > 0:
      throttle = min(acceleration / max_accel, 1.0)
      brake = 0.0
  else:
      throttle = 0.0
      brake = min(-acceleration / max_brake, 1.0)
  ```
* Consider PID controller for smoother throttle control

### **Lead Vehicle Control**
Two approaches:
1. **Traffic Manager:** Set target velocity dynamically
2. **Custom Control:** Manually update velocity following predefined trajectory

### **Common Issues & Solutions**
* **Connection timeout:** Increase client timeout: `client.set_timeout(10.0)`
* **Spawn failures:** Check for collision-free spawn points
* **Jerky control:** Reduce dt or improve acceleration-to-control mapping
* **Memory leaks:** Ensure all actors destroyed, restart server periodically
* **Slow training:** Use -RenderOffScreen flag, reduce quality-level

### **Testing Workflow**
1. Start CARLA server
2. Test environment connection with simple script
3. Verify vehicle spawning and control
4. Test single MPC controller before DRL integration
5. Run short training test (1000 steps) before full training
6. Monitor for errors and adjust parameters

### **Performance Tips**
* Use Town01 or Town03 (simpler maps) for faster simulation
* Disable unnecessary sensors
* Reduce physics sub-steps if accuracy permits
* Consider running multiple CARLA instances on different ports for parallel training