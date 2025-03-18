from stable_baselines3 import PPO, DQN, TD3
from stable_baselines3.common.type_aliases import GymEnv

from src.rl.models.policies import getPolicy

algo = ["PPO", "DQN", "TD3", "DQNLSTM"]

def getRLAlgo(
    env: GymEnv, 
    test_name: str, 
    model_config: dict
):
    algo_name = model_config['rl_algo']
    print(model_config)
    policyNetwork = getPolicy(model_config['policy'])
    
    if algo_name == algo[0]:
        return PPO(
            policyNetwork, 
            env, 
            verbose=model_config['verbose'], 
            learning_rate=model_config['learning_rate'],
            gamma=model_config['gamma'],
            batch_size=model_config['batch_size'],  
            n_steps=model_config['n_steps'], 
            ent_coef=model_config['ent_coef'], 
            clip_range=model_config['clip_range'],  
            max_grad_norm=model_config['max_grad_norm'],
            tensorboard_log=f'./logs/{test_name}/'
        )
    elif algo_name == algo[1]:
        return DQN(
            policyNetwork,
            env,
            verbose=1,
            learning_rate=model_config['learning_rate'],
            gamma=model_config['learning_rate'],
            batch_size=model_config['batch_size'],  # Increased batch size
            buffer_size=model_config['buffer_size'],  # Adjusted buffer size
            learning_starts=model_config['learning_starts'],  # Reduced learning starts
            target_update_interval=model_config['target_update_interval'],  # Adjusted target update interval
            train_freq=model_config['train_freq'],
            gradient_steps=model_config['gradient_steps'],
            exploration_fraction=model_config['exploration_fraction'],  # Increased exploration fraction
            exploration_initial_eps=model_config['exploration_initial_eps'],
            exploration_final_eps=model_config['exploration_final_eps'],  # Reduced final exploration epsilon
            max_grad_norm=model_config['max_grad_norm'],
            tensorboard_log=f'./logs/{test_name}/',  
        )
    
    elif algo_name == algo[2]:
        return TD3(
            policyNetwork,
            env,
            learning_rate=0.001,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.999,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            action_noise=None,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            tensorboard_log=f'./logs/{test_name}/',
            verbose=1,
            device="auto",
            _init_setup_model=True
        )
        
    
    elif algo_name == algo[3]:
        policy_kwargs = dict(
            hidden_size=64,
            lstm_layers=1,
            sequence_length=8  # Specify sequence length
        )

        return DQN(
            policyNetwork,
            env,
            verbose=1,
            learning_rate=model_config['learning_rate'],
            gamma=model_config['learning_rate'],
            batch_size=model_config['batch_size'],  # Increased batch size
            buffer_size=model_config['buffer_size'],  # Adjusted buffer size
            learning_starts=model_config['learning_starts'],  # Reduced learning starts
            target_update_interval=model_config['target_update_interval'],  # Adjusted target update interval
            train_freq=model_config['train_freq'],
            gradient_steps=model_config['gradient_steps'],
            exploration_fraction=model_config['exploration_fraction'],  # Increased exploration fraction
            exploration_initial_eps=model_config['exploration_initial_eps'],
            exploration_final_eps=model_config['exploration_final_eps'],  # Reduced final exploration epsilon
            max_grad_norm=model_config['max_grad_norm'],
            tensorboard_log=f'./logs/{test_name}/',
            policy_kwargs=policy_kwargs  
        )

    
    raise ValueError("Wrong algorithm name")

def getRLType(algo_name: str):
    if algo_name == "PPO":
        return PPO
    elif algo_name == "DQN":
        return DQN
    
    elif algo_name == "TD3":
        return TD3
    
    elif algo_name == "DQNLSTM":
        return DQN

    raise ValueError("Wrong RL algo name")

