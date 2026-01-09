import gymnasium as gym
from gymnasium import spaces
import numpy as np
import genesis as gs
import torch
from scipy.spatial.transform import Rotation as R

class LungEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array", device="cpu", continuous_action=True, debug=False):
        super().__init__()
        
        self.render_mode = render_mode
        self.device = device
        self.continuous_action = continuous_action
        self.debug = debug
        
        # --- Genesis Init ---
        # Initialize genesis if not already done. 
        # gs.init might throw if called twice, or it might be idempotent. 
        # For safety, we assume the user/script calls gs.init(). 
        # If strictly needed here:
        # try:
        #     gs.init(backend=gs.cpu if device=="cpu" else gs.gpu, precision="32", logging_level="warning")
        # except Exception:
        #     pass

        # --- Action Space ---
        # 1. Motion: 
        #    If continuous: [vx, vy, vz, wy, wp, wr] (6 DOF) or subset. 
        #    Let's assume 6DOF relative control in camera frame.
        #    Range: [-1, 1] mapped to max velocities.
        
        # 2. Estimation:
        #    Pose: [x, y, z, qw, qx, qy, qz] (7 dims) - Estimate of Lung Pose
        #    Deformation: [actuation_magnitude] (1 dim) - Estimate of breathing state
        #    Total Estimation: 8 dims.
        
        # Composite Action Space
        if self.continuous_action:
            # Motion: 6 dims
            motion_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        else:
             # Discrete: 0=NoOp, 1= Fwd, 2=Back, 3=Left, 4=Right, 5=Up, 6=Down...
             motion_space = spaces.Discrete(13) 

        # Estimation space: Unbounded (or loosely bounded) reals
        estimation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.action_space = spaces.Dict({
            "motion": motion_space,
            "estimation": estimation_space
        })

        # --- Observation Space ---
        # 1. Visual: RGB (H, W, 3)
        self.img_size = (1280, 960) # Default smaller size for RL
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            # Optional: Add depth if needed
            # "depth": ...
        })

        # --- Simulation Params ---
        self.scene = None
        self.lungs = None
        self.bronchi = None
        self.cam = None
        self.initial_state = None
        
        # Tracking internal state
        self._step_count = 0
        self._max_steps = 200
        
        # Initialize the scene once (Genesis architecture prefers one Scene object)
        self._build_scene(render_mode)

    def _build_scene(self):
        # Create Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=2e-3,
                substeps=10,
                gravity=(0, 0, -9.8),
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -0.5, -0.4),
                upper_bound=(0.5, 0.9, 1.0),
                grid_density=32,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=self.debug, # cleaner look
                show_world_frame=self.debug,
            ),
            show_viewer= self.render_mode == "human", # Headless for RL usually
        )
        
        # Ground plane (Optional, might distract from lungs)
        # self.scene.add_entity(morph=gs.morphs.Plane())

        # Lungs - MPM Elastic
        self.lungs = self.scene.add_entity(
            material=gs.materials.MPM.Muscle(
                E=5e3, 
                nu=0.4, 
                rho=500.0,
            ),
            morph=gs.morphs.Mesh(
                file="assets/lung_lobes.obj",
                pos=(0.0, 0.0, 0.25),
                scale=0.2,
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.9, 0.6, 0.6, 0.8),
                vis_mode="visual",
            ),
        )

        # Bronchi - MPM Elastic (Fixed/Rigid-like but MPM for coupling)
        # Note: Using MPM.Elastic as requested by user correction
        self.bronchi = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                 # stiffness can be higher to simulate cartilage rigidity
                 E=5e4,
                 nu=0.2,
                 rho=1000.0,
            ),
            morph=gs.morphs.Mesh(
                file="assets/bronchi.obj",
                pos=(0.0, 0.0, 0.2), 
                scale=0.2,
                euler=(0, 0, 0),
                fixed=False,
            ),
            surface=gs.surfaces.Default(
                color=(0.8, 0.7, 0.6, 1.0),
            ),
        )

        # Camera Agent
        self.cam = self.scene.add_camera(
            res=self.img_size,
            pos=(0.65, -0.45, 0.6),
            lookat=(0.0, 0.0, 0.3),
            fov=60, # Wider FOV for inside view
            GUI=False,
        )
        
        self.scene.build()
        self.initial_state = self.scene.get_state()
        
        # Keep track of the camera state manually as Genesis camera is kinematic
        self.cam_pos = np.array([0.65, -0.45, 0.6])
        self.cam_lookat = np.array([0.0, 0.0, 0.3])

    def render(self):
        if self.render_mode == "rgb_array":
            return self.cam.render(rgb=True, depth=False)[0]
        elif self.render_mode == "human":
            self.scene.render()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Reset Physics
        # Note: Proper MPM reset requires clearing dynamic nodes as seen in sample code
        if hasattr(self.scene, "_visualizer") and self.scene._visualizer:
             self.scene._visualizer._context.clear_dynamic_nodes(only_outdated=False)
        
        self.scene.reset(state=self.initial_state)
        
        if self.scene.mpm_solver.is_active:
             self.scene.mpm_solver.update_render_fields()
        
        # 2. Domain Randomization (Episodic)
        # Randomize Lung/Bronchi Pose
        # For MPM entities, we manipulate particles.
        
        # A. Random Rotation
        # generate_random_rotation_matrix(1) from ref uses utils.
        # We can just use scipy.
        random_rot = R.random().as_matrix() # 3x3
        center = np.array([0.0, 0.0, 0.3]) # Approximate center
        
        # Apply to entities
        self._rotate_mpm_entity(self.lungs, random_rot, center)
        self._rotate_mpm_entity(self.bronchi, random_rot, center)
        
        # Store GT for reward
        # GT Pose: We define the "Pose" as the translation/rotation applied to the canonical mesh
        # Here we just applied 'random_rot'. Translation might be 0 for now or randomized too.
        quat = R.from_matrix(random_rot).as_quat(scalar_first=True) # w,x,y,z
        self.gt_pose = np.concatenate([center, quat]) # [x,y,z, w,x,y,z] roughly
        self.gt_deformation = np.array([0.0]) # Initialize
        
        # Reset Camera to random initial position nearby? Or fixed start?
        # Let's start fixed for now.
        self.cam_pos = np.array([0.65, -0.45, 0.6])
        self.cam_lookat = np.array([0.0, 0.0, 0.3])
        self.cam.set_pose(pos=self.cam_pos, lookat=self.cam_lookat)

        self._step_count = 0
        
        return self._get_obs(), {
            "gt_pose": self.gt_pose,
            "gt_deformation": np.array([0.0]) # Initial deformation is 0 or random?
        }

    def step(self, action):
        motion_act = action["motion"]
        est_act = action["estimation"]
        
        # 1. Apply Motion (Relative Camera Control)
        self._apply_camera_motion(motion_act)
        
        # 2. Apply Deformation (Breathing)
        # Simple cyclic actuation for now
        t = self._step_count * self.scene.sim_options.dt
        actuation_val = 0.5 * (0.5 + np.sin(2.0 * np.pi * t)) # Mock frequency
        self.lungs.set_actuation(np.array([actuation_val]))
        self.gt_deformation = np.array([actuation_val])
        
        # 3. Step Physics
        self.scene.step()
        
        # 4. Get Obs
        obs = self._get_obs()
        
        # 5. Compute Reward
        reward = self._compute_reward(est_act)
        
        # 6. Check Done
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self._max_steps
        
        info = {
            "gt_pose": self.gt_pose,
            "gt_deformation": self.gt_deformation
        }
        
        return obs, reward, terminated, truncated, info

    def _apply_camera_motion(self, action):
        # Motion logic: Relative to Camera Frame
        
        # Calculate View Vector
        view_vec = self.cam_lookat - self.cam_pos
        dist = np.linalg.norm(view_vec)
        if dist < 1e-6: dist = 1.0
        
        fwd = view_vec / dist
        
        # Arbitrary Up convention (World Z usually, assumes camera isn't fully upside down)
        world_up = np.array([0,0,1])
        right = np.cross(fwd, world_up)
        # Normalize right
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
             # Singularity (looking straight up/down)
             # Fallback
             right = np.array([1,0,0])
        else:
             right /= right_norm
             
        real_up = np.cross(right, fwd)
        
        # Action: [vx, vy, vz, ...] 
        speed_scale = 0.05
        
        if self.continuous_action:
            # action in Box(-1, 1)
            # Translation
            # action[0]=Right(x), action[1]=Up(y), action[2]=Fwd(z)
            # Note: typical camera coordinates: X right, Y down (or up), Z fwd. 
            # Let's map X=right, Y=up, Z=fwd
            
            # Use clipping to ensure safety if action is out of bounds
            action = np.clip(action, -1.0, 1.0)
            
            delta_p = (action[0] * right + 
                       action[1] * real_up + 
                       action[2] * fwd) * speed_scale
            
            # Apply Translation
            self.cam_pos += delta_p
            # Move lookat with pos to maintain relative view direction (strafe)
            self.cam_lookat += delta_p
            
            # Rotation (Yaw/Pitch)
            # action[3]=Yaw (around real_up), action[4]=Pitch (around right)
            rot_speed = 0.05
            yaw = action[3] * rot_speed
            pitch = action[4] * rot_speed
            
            # Apply rotations to the forward vector
            # Yaw
            if abs(yaw) > 1e-6:
                 r_yaw = R.from_rotvec(real_up * -yaw) # Negative for standard look behavior
                 fwd = r_yaw.apply(fwd)
                 # Recompute right/up after yaw
                 right = np.cross(fwd, world_up)
                 right /= (np.linalg.norm(right) + 1e-6)
                 real_up = np.cross(right, fwd)

            # Pitch
            if abs(pitch) > 1e-6:
                 r_pitch = R.from_rotvec(right * pitch)
                 fwd = r_pitch.apply(fwd)
            
            # Reconstruct lookat
            self.cam_lookat = self.cam_pos + fwd * dist

        else:
            # Discrete implementation
            # 0: Noop
            # 1: Fwd, 2: Back...
            pass
            
        self.cam.set_pose(pos=self.cam_pos, lookat=self.cam_lookat)

    def _get_obs(self):
        # Genesis Render
        # self.cam.render() # Not strictly needed if render(args) calls it, but safe.
        rgb, _, _, _ = self.cam.render(rgb=True, depth=False)
        
        # Ensure format
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
            
        # Shape might be (H,W,4) for RGBA or (H,W,3)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
            
        # Convert to uint8 0-255 if float 0-1
        # Check dtype
        if rgb.dtype in [np.float32, np.float64]:
             rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            
        return {"image": rgb}

    def _compute_reward(self, est_action):
        # est_action: [8 dims]
        # est_pose (7), est_def (1)
        
        pred_pose = est_action[:7]
        pred_def = est_action[7:]
        
        # 1. Pose Loss (Pose estimation is hard, let's look at translation only first or both)
        # GT: [cx, cy, cz, qw, qx, qy, qz]
        
        # Translation dist
        t_dist = np.linalg.norm(pred_pose[:3] - self.gt_pose[:3])
        
        # Rot dist (Quaternion dot product)
        # Handle antipodal equality (q == -q)
        q_dot = np.abs(np.dot(pred_pose[3:], self.gt_pose[3:]))
        r_dist = 1.0 - q_dot  # 0 if same, 1 if wrong
        
        # Def Loss
        d_dist = np.linalg.norm(pred_def - self.gt_deformation)
        
        # Weighted sum
        loss = 1.0 * t_dist + 0.5 * r_dist + 1.0 * d_dist
        
        return -loss

    def _rotate_mpm_entity(self, entity, rotation_matrix, center=None):
        # Helper to rotate particles
        current_positions = entity.get_particles_pos() 
        # Check backend
        is_torch = isinstance(current_positions, torch.Tensor)
        
        if is_torch:
            rm = torch.tensor(rotation_matrix, device=current_positions.device, dtype=current_positions.dtype)
            if center is not None:
                c = torch.tensor(center, device=current_positions.device, dtype=current_positions.dtype)
            else:
                c = torch.mean(current_positions, dim=0)
            
            rel = current_positions - c
            rot_rel = torch.matmul(rel, rm.T) # (N,3) @ (3,3).T = (N,3)
            new_pos = c + rot_rel
            
            entity.set_particles_pos(new_pos)
            entity.set_particles_vel(torch.zeros_like(new_pos))
        else:
            # Numpy fallback (unlikely in this context but good practice)
            pass
