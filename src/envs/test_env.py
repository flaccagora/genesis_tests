import gymnasium as gym
import numpy as np
import genesis as gs
import matplotlib.pyplot as plt
import src.envs # This registers the env

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", choices=["rgb_array", "human", "none"])
    args = parser.parse_args()
    
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    
    # Create env
    env = gym.make('GenesisLung-v0', render_mode=args.render_mode, device='cuda')
    
    print("Environment created.")
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    
    # Reset
    obs, info = env.reset()
    print("Reset complete. Info keys:", info.keys())
    print("Initial GT Pose:", info['gt_pose'])
    
    # Save initial observation
    if 'image' in obs:
        plt.imsave('debug_init_obs.png', obs['image'])
        print("Saved debug_init_obs.png")
    
    # Step loop
    for i in range(1000):
        # Random action
        motion_act = env.action_space['motion'].sample()
        
        # Fake estimation (perfect gt + noise)
        gt_pose = info['gt_pose']
        gt_def = info['gt_deformation']
        
        # Flat estimation vector
        est_act = np.concatenate([gt_pose, gt_def]) + np.random.normal(0, 0.1, size=8)
        
        action = {
            "motion": motion_act,
            "estimation": est_act.astype(np.float32)
        }
        
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i}: Reward={reward:.4f}")
        
    # Save last frame
    if 'image' in obs:
        plt.imsave('debug_step_10.png', obs['image'])
        print("Saved debug_step_10.png")

    env.close()

if __name__ == "__main__":
    main()
