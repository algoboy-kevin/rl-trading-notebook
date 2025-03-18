from .lstmqnet import LSTMDQNPolicy

policy = ['MlpPolicy', 'DQNLSTM']

def getPolicy(policy_name: str):
    if policy_name == 'MlpPolicy':
        return 'MlpPolicy'
    
    if policy_name == "DQNLSTM":
        return LSTMDQNPolicy
    
    raise ValueError("Wrong policy name")