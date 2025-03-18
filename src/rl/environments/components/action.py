import numpy as np
from gymnasium.spaces import Dict, Discrete, Box
actions= ["default", "box"]

def checkActionName(action_key: str):
    if action_key in actions:
        return
    
    raise ValueError("Wrong obs config name")

def getActionSpace(action_key: str):
    if action_key == "default":
        return Discrete(3)
    
    if action_key == "box":
        return Box(-1, 1, (1,), dtype=np.float32)
    
def getActionFromBox(action):
    if action > 0.33:
        return 0
    
    if action < -0.33:
        return 1
    
    return 2
