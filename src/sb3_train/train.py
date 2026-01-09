import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import src.envs  # Registers GenesisLung-v0

class FlattenActionWrapper(gym.ActionWrapper):
    """
    Flattens the Dict action space {motion: (6,), estimation: (8,)} 
    into a single Box(14,) for SB3 compatibility.
    """
    def __init__(self, env):
        super().__init__(env)
        # Assuming both are Box spaces based on current implementation
        # Motion: (6,)
        # Estimation: (8,)
        
        # We need to know the shapes / limits
        # Accessing inner spaces
        motion_space = self.env.action_space["motion"]
        est_space = self.env.action_space["estimation"]
        
        # Calculate total size
        # Handling Discrete motion is tricky with flattening, lets assume Continuous for PPO default
        # If motion is discrete, we would have MultiDiscrete or similar.
        # Current LungEnv default is continuous_action=True.
        
        if isinstance(motion_space, spaces.Discrete):
             raise NotImplementedError("Wrapper currently only supports continuous motion action space.")
             
        self.motion_dim = motion_space.shape[0]
        self.est_dim = est_space.shape[0]
        total_dim = self.motion_dim + self.est_dim
        
        # Define new action space
        # Motion is [-1, 1], Est is [-inf, inf]. 
        # SB3 likes normalized actions usually, but we can define bounds.
        # SB3 requires finite bounds. 
        # Motion is [-1, 1].
        # Estimation is unbounded in theory, but we must set limits.
        # Pose is small (meters), Quat is -1..1. Params are 0..1.
        # using +/- 100 is safe.
        low = np.full((total_dim,), -100.0, dtype=np.float32)
        high = np.full((total_dim,), 100.0, dtype=np.float32)
        
        # Set motion bounds specifically
        low[:self.motion_dim] = -1.0
        high[:self.motion_dim] = 1.0
        
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def action(self, action):
        # action is float32 array (14,)
        motion_act = action[:self.motion_dim]
        est_act = action[self.motion_dim:]
        
        return {
            "motion": motion_act,
            "estimation": est_act
        }


def make_env():
    # Helper for VecEnv
    env = gym.make("GenesisLung-v0", render_mode="rgb_array", continuous_action=True)
    env = FlattenActionWrapper(env)
    return env

import genesis as gs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--check_freq", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(backend=gs.cpu if args.device == "cpu" else gs.gpu, precision="32", logging_level="warning")

    # Create Log Dir
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create Env
    # Use DummyVecEnv for simpler debugging or Subproc for speed
    # Genesis might have issues with Subproc if GPU context is shared or limited?
    # Let's stick to DummyVecEnv for safety first.
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=DummyVecEnv)

    # Initialize Agent
    # CnnPolicy for images
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
        batch_size=64,
        n_steps=2048,
    )

    print(f"Starting training on {args.device} for {args.steps} steps...")
    
    # Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.check_freq,
        save_path=args.log_dir,
        name_prefix="lung_model"
    )

    # Train
    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    
    print("Training finished.")
    model.save(os.path.join(args.log_dir, "final_model"))
    
    vec_env.close()

if __name__ == "__main__":
    main()
