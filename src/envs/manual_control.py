import gymnasium as gym
import numpy as np
import genesis as gs
import sys
import time
from pynput import keyboard

import src.envs # Registers env

class ManualAgent:
    def __init__(self, env):
        self.env = env
        self.pressed_keys = set()
        self.running = True
        
        # Action mappings
        # Motion: [vx, vy, vz, yaw, pitch, roll]
        # v: Right, Up, Forward
        
        # WASD - Plane movement (Forward/Left/Back/Right) - Typical FPS
        # W: Forward (+Z)
        # S: Backward (-Z)
        # A: Left (-X)
        # D: Right (+X)
        
        # Q/E - Up/Down (Y)
        # Q: Up (+Y)
        # E: Down (-Y)
        
        # Arrows - Rotation
        # Left/Right: Yaw
        # Up/Down: Pitch
        
        self.key_map = {
            'w': (2, 1.0),   # Forward
            's': (2, -1.0),  # Backward
            'd': (0, 1.0),   # Right
            'a': (0, -1.0),  # Left
            'q': (1, 1.0),   # Up
            'e': (1, -1.0),  # Down
        }
        
        self.rot_map = {
            keyboard.Key.left:  (3, 1.0),  # Yaw Left
            keyboard.Key.right: (3, -1.0), # Yaw Right
            keyboard.Key.up:    (4, 1.0),  # Pitch Up
            keyboard.Key.down:  (4, -1.0), # Pitch Down
        }

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            return False
            
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.add(key.char.lower())
            else:
                self.pressed_keys.add(key)
        except AttributeError:
            self.pressed_keys.add(key)

    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.discard(key.char.lower())
            else:
                self.pressed_keys.discard(key)
        except AttributeError:
            self.pressed_keys.discard(key)

    def get_action(self):
        # Construct action vector
        # Motion: 6 dims
        motion = np.zeros(6, dtype=np.float32)
        
        for k, (idx, val) in self.key_map.items():
            if k in self.pressed_keys:
                motion[idx] += val
                
        for k, (idx, val) in self.rot_map.items():
            if k in self.pressed_keys:
                motion[idx] += val
                
        # Estimation: 8 dims (0 for now)
        estimation = np.zeros(8, dtype=np.float32)
        
        return {
            "motion": motion,
            "estimation": estimation
        }

def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    
    env = gym.make('GenesisLung-v0', render_mode='human', continuous_action=True, debug=True)
    env.reset()
    
    agent = ManualAgent(env)
    
    print("="*60)
    print("MANUAL CONTROL STARTED")
    print("Controls:")
    print("  W/S: Forward/Backward")
    print("  A/D: Left/Right")
    print("  Q/E: Up/Down")
    print("  Arrows: Look (Yaw/Pitch)")
    print("  ESC: Exit")
    print("="*60)
    
    try:
        while agent.running:
            action = agent.get_action()
            
            # Step env
            obs, reward, term, trunc, info = env.step(action)
            
            # Render happens automatically in 'human' mode in step() usually, 
            # OR we call it manually if the env requires it.
            # LungEnv.render('human') calls scene.render() which updates viewer.
            # Usually step() logic calls render if mode is human, or validGym envs 
            # expect user to call env.render().
            # Let's call env.render()
            env.render()
            
            # Sleep to cap FPS approx (e.g. 60 FPS = 0.016s)
            # Genesis might block on VSync?
            time.sleep(0.01)
            
            if term or trunc:
                env.reset()
                
    except KeyboardInterrupt:
        pass
    finally:
        agent.listener.stop()
        env.close()

if __name__ == "__main__":
    main()
