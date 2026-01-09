# Gymnasium Environment Specification: Genesis Lung Inspection

## 1. Overview
This specification defines a custom **Gymnasium** reinforcement learning environment (`LungEnv`) designed for **Active Perception** inside a deformable lung simulation. The agent controls a camera (bronchoscope) to navigate and simultaneously estimate the pose and deformation parameters of the lung. The physics and rendering are powered by **Genesis**.

## 2. World Model & Simulation (Genesis)
The environment represents a dynamic, deformable biological scene.

*   **Physics Engine**: Genesis (Python/C++ backend).
*   **Material Types**:
    *   **Lungs**: MPM (Material Point Method) Elastic material. Represents soft tissue that deforms cyclically.
    *   **Bronchi**: MPM (Mesh). Coupled with the lung MPM particles to simulate structural interaction.
*   **Scene Dynamics**:
    *   **Breathing**: Simulated via actuation parameters applied to the MPM inputs (or boundary conditions), creating cyclic deformations.
    *   **Coupling**: Rigid bronchi move/rotate; soft lung tissue follows or reacts to collisions.

### Domain Randomization (Episodic)
To ensure robust SIM2REAL transfer, the environment randomizes critical parameters at the start of every episode (`reset()`):
1.  **Lung Pose**: Random Rotation ($R \in SO(3)$) and Translation ($t \in \mathbb{R}^3$) of the entire lung/bronchi complex.
2.  **Lung Morphology**:
    *   Variation in global scale.
    *   (Future) Selection from a dataset of different patient meshes.
3.  **Deformation Dynamics**:
    *   Randomized amplitude and frequency of the "breathing" actuation.
4.  **Bronchi Configuration**:
    *   Randomized relative placement or orientation of the bronchi within the lung tissue.
5.  **Visuals** (Optional but recommended):
    *   Randomized lighting intensity/position.
    *   Texture variations (mucosa color, specularity).

## 3. The Agent (Bronchoscope)
The agent acts as an **Active Observer**. It does not physically manipulate the tissue (initially) but moves to gather information.

*   **Embodiment**: Free-flying camera (6-DoF or Constrained 4-DoF).
*   **Reference Frame**: All movement actions are **Relative to the Camera Frame** (First-Person View).
    *   *Forward* is along the camera's optical axis.

## 4. Observation Space
The agent receives raw sensory data matching a real endoscopic feed.

*   **Visual**:
    *   **RGB**: $H \times W \times 3$ (uint8 or float).
    *   **Depth**: $H \times W \times 1$ (float).
    *   *Note*: Point clouds are NOT provided directly to the agent to enforce visual learning, though they exist in ground truth for verification.
*   **Proprioception** (Optional):
    *   Current Camera Pose (if the scope has EM tracking).
    *   Last Action taken.

## 5. Action Space
The environment supports a **Composite Action Space**, requiring the agent to effectively multi-task: **Move** and **Estimate**.

$$ A_t = \{ A_{motion}, A_{estimation} \} $$
*   **Flattened Action**: For standard RL libraries (SB3), the dictionary is often flattened or handled via `MultiInputPolicy`. The implementation uses a `Dict` action space, but standard algorithms like PPO/SAC in SB3 can handle `Dict` observations. For `Dict` *actions*, SB3 is limited. Thus, we may flatten this to a single Box space `[motion... estimation...]` for training or use a wrapper.

### A. Motion Control ($A_{motion}$)
Configurable as either **Continuous** or **Discrete** via initialization arguments.
*   **Relative control in Camera Frame**:
    *   *Linear*: $v_x$ (Right), $v_y$ (Up), $v_z$ (Forward).
    *   *Angular*: $\omega_{yaw}, \omega_{pitch}, \omega_{roll}$.
*   **Modes**:
    *   `Box(-1, 1)`: Continuous velocity commands.
    *   `Discrete(N)`: Fixed step increments (e.g., [Move Forward, Turn Left, Stop...]).

### B. Estimation Output ($A_{estimation}$)
At every timestep, the agent must output its current belief of the hidden state.
*   **Pose Estimate ($\hat{P}$)**: Predicted Position ($x,y,z$) and Orientation (Quaternion $q_w, q_x, q_y, q_z$) of the lungs.
*   **Deformation Estimate ($\hat{D}$)**: Predicted parameters of the breathing cycle (e.g., `[amplitude, phase, frequency]`).

## 6. Reward Architecture
The reward function is **modular** and **user-definable**, supporting End-to-End training.

### Default Rewards
1.  **Estimation Error (Dense)**:
    $$ R_{est} = - \alpha \| P_{GT} - \hat{P} \| - \beta \| D_{GT} - \hat{D} \| $$
    *   Penalizes deviation from the ground truth pose and deformation parameters.
2.  **Navigation/Coverage (Optional)**:
    *   Reward for keeping the camera centered in the airway.
    *   Reward for maximizing visible surface area (View Entropy).

### Customization
The environment accepts a `Function` or `Class` to calculate rewards:
```python
def custom_reward(obs, state, info):
    # User implements arbitrary logic here
    return calculated_reward
```

## 7. Technical Requirements
*   **API**: Standard Gymnasium (`gym.Env`).
*   **Rendering**: Differentiable/Fast rendering via Genesis (Rasterizer or Raytracer).
*   **Performance**: Should support parallel environments for efficient RL training.

## 8. Training Pipeline (Stable Baselines 3)
*   **Library**: Stable Baselines 3 (SB3).
*   **Algorithm**: **PPO** (Proximal Policy Optimization) or **SAC** (Soft Actor-Critic).
*   **Policy Architecture**:
    *   **Visual Encoder**: CNN (NatureCNN or ResNet-like) to process RGB observations.
    *   **Action Head**: Outputs combined vector for [Motion (6) + Estimation (8)].
*   **Infrastructure**:
    *   `src/sb3_train/`: Contains training scripts and utilities.
    *   **Wrappers**:
        *   `FlattenActionWrapper`: Converts `Dict` actions to `Box` for compatibility if needed.
        *   `Monitor`: Tracks episode rewards/lengths.