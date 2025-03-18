import argparse
import os
from typing import Dict

from .rl.environments import TradingEnvironment
from .rl.models import TradingModel
from .rl.libs import doesCheckpointExist, getLastEpisode, ConfigManager

localDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train(config: ConfigManager, saveEvery: int, maxStep: int, fileName: str, directory: str):
    print(f"Training case: {config.config['test_name']}")
    checkpointPath = os.path.join(localDir, f"models/{config.config['test_name']}")
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)

    env = TradingEnvironment(
        fileName,
        directory,
        config.config['env_config']
    )

    model = TradingModel(
        env, 
        config.config['test_name'],
        config.config['model_config']
    )

    # check if checkpoint exists
    print("Finding checkpoint on:", checkpointPath)
    isExist = doesCheckpointExist(checkpointPath)

    print("Exist?", isExist)
    if isExist:
       lastCheckpoint = getLastEpisode(checkpointPath)
       model.load(lastCheckpoint, env)
    
    iter_completed = 0

    try:
        while model.model.num_timesteps < maxStep:
            model.train(
                total_timesteps=saveEvery
            )
            total_timesteps = model.model.num_timesteps
            model.save(f"{checkpointPath}/checkpoint-{total_timesteps}")
            iter_completed += 1
            print(f"Model saved at {total_timesteps} timesteps")

        # Final save when reaching 2 million timesteps
        if iter_completed > 0:
            model.save(f"{checkpointPath}/checkpoint-{model.model.num_timesteps}")
            print("Training completed. Final model saved.")
        else:
            print(f"Already above maximum step. Model ts: {model.model.num_timesteps}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        total_timesteps = model.model.num_timesteps
        model.save(f"{checkpointPath}/checkpoint-{total_timesteps}")
        print(f"Model saved to {checkpointPath}/checkpoint-{total_timesteps}")


# def main(saveEvery: int, maxStep: int, filename: str, configFile: str):
#     configPath = os.path.join(localDir, f"configs/{configFile}")
#     config = ConfigManager(configPath)
#     train(config, saveEvery, maxStep, filename)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train the trading model.")
#     parser.add_argument("--filename", type=str, help="Name of the file price reference.")
#     parser.add_argument("--config", type=str, help="Name of the configuration file.")
#     parser.add_argument("--multiple", action="store_true", help="Run multiple configurations")

#     args = parser.parse_args()


#     if args.multiple == True:
#         if not args.filename:
#             raise ValueError("Both --filename arguments must be provided.")
    
#         print(args)
#         run_multiple(args.filename)

#     else:
#         # Check if filename and config are provided
#         if not args.filename or not args.config:
#             raise ValueError("Both --filename and --config arguments must be provided.")

#         main(100000, 13e6, args.filename, args.config)