import numpy as np

from typing import Any, SupportsFloat
from gymnasium import Env
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict
from .components import (
    Broker, 
    PriceProvider, 
    getActionSpace,  
    RewardCounter,
    ObservationProvider,
    ParamsObsDualMA,
    ParamsObsDonchian,
    ParamsPostActionDonchian,
    ParamsPostUpdateDonchian,
    ParamsPostActionDualMA,
    ParamsPostUpdateDualMA,
    get_donchian_obs_dict
    )
from src.rl.libs.utils import OrderInfo, available_strategy

class TradingEnvironment(Env):
    def __init__(
        self, 
        file_name: str,
        directory: str,
        env_config: Dict,
        record_history = False,

        ):
        super(TradingEnvironment, self).__init__()
        self.action_space = getActionSpace(env_config['action'])  # Buy, Sell, Hold
        self.reward_counter = RewardCounter(env_config['strategy_type'])
        self.observation_provider = ObservationProvider(env_config['strategy_type'])
        self.observation_space = self.observation_provider.get_observation_space()
        self.action_type = env_config['action']
        self.terminated = False
        self.truncated = False
        self.context = {}
        self.broker = Broker(
            env_config,
            record_history,
        )

        self.price_provider = PriceProvider(
            file_name, 
            directory,
            env_config['is_random'],
            env_config['indicators'],
            env_config['derived_indicators']
        )
        self.current_step = 0

    def reset(self, seed= 1, options={}) -> tuple[ObsType, dict[str, Any]]:
        self.current_step = 0
        self.reward_counter.reset()
        self.context = {}
        self.price_data = self.price_provider.fetchDataSlice(4000)

        row_data = self.price_data[self.current_step]
        self.broker.reset(row_data)
        self.truncated = False
        self.terminated = False
        self.initial_price = row_data['SMA5']

        obs = self.get_observation()
        info = {}

        # Reset the environment to the starting state. Should return observation and info (optional)
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if 'obs' not in self.context:
            self.context['obs'] = self.get_observation()

        # t-0 - Before next step
        self.pre_action_context_update()
        
        # t-0 Execute action
        info = self.broker.executeAction(action)
        self.post_action_context_update(info, action)
        
        # t-0 Evaluate before passing to the next state
        step_rewards = self.calculate_post_action_reward()
        
        # t+1 The next 5 minutes, update to next row and data
        self.current_step += 1
        row_data = self.price_data[self.current_step]
        self.broker.updateState(row_data)

        # t+1 Record observation before limit and hazard evaluation
        obs = self.get_observation()
        self.context.update({
            "obs": obs,
        })

        # t+1 Check for limitation and hazard
        self.evaluate_limit_hazard()

        info = {}

        # t+1 Rewards calculationstep
        step_rewards += self.calculate_post_update_reward()
        
        self.broker.reward_cum += step_rewards
        self.check_if_terminated()

        return obs, step_rewards, self.terminated, self.truncated, info

    def render(self):
        # Optionally implement rendering
        pass

    def calculate_post_action_reward(self):
        reward = int(0)

        if self.reward_counter.type == available_strategy[0]:
            params = ParamsPostActionDualMA(
                action=self.context['action'],
                info=self.context['info'],
                v3=self.broker.row_data['V3'],
                inventory=self.broker.im.inventory,
                mark_price=self.broker.row_data['SMA5'],
                avg_position=self.broker.im.getAveragePrice('SMA5'),
                lowest_position=self.broker.lowest_point,
                last_buy_position=self.broker.im.getLastEntry('SMA5')
            )

            return self.reward_counter.calculate_post_action(params)

        if self.reward_counter.type == available_strategy[1]:
            obs = get_donchian_obs_dict(self.context['obs'])
            
            params = ParamsPostActionDonchian(
                action=self.context['action'],
                info=self.context['info'],
                inventory=self.broker.im.inventory,
                obs=obs
            )

            return self.reward_counter.calculate_post_action(params)

        return reward

    def calculate_post_update_reward(self):
        reward = int(0)

        if self.reward_counter.type == available_strategy[0]:
            params = ParamsPostUpdateDualMA(
                ma_current= self.price_data[self.current_step]['SMA5'],
                ma_before=self.price_data[self.current_step - 1]['SMA5'],
                i_post_action=self.broker.im.inventory
            )

            reward = self.reward_counter.calculate_post_update(params)

        if self.reward_counter.type == available_strategy[1]:
            params = ParamsPostUpdateDonchian(
                ma_current= self.price_data[self.current_step]['SMA5'],
                ma_before=self.price_data[self.current_step - 1]['SMA5'],
                i_post_action=self.broker.im.inventory,
            )

            reward = self.reward_counter.calculate_post_update(params)

        return reward
    
    def evaluate_limit_hazard(self):
        maxLenEps = self.current_step >= len(self.price_data) - 1
        maxDD = self.broker.getMaxDD() > 0.5

        if maxLenEps or maxDD:
            self.terminated = True

        if self.terminated:
            self.broker.closeAllOrders()


    def pre_action_context_update(self):
        pass
      

    def post_action_context_update(self, info: OrderInfo, action: int):
        self.context.update({
            "action": int(action),
            "info": info
        })

    def get_observation(self):
        params = {}

        if self.observation_provider.type == available_strategy[0]:
            n_long, n_short = self.broker.im.getInventoryCount()
            params = ParamsObsDualMA(
                row_data=self.broker.row_data,
                mark_price=self.broker.row_data['SMA5'],
                avg_position=self.broker.im.getAveragePrice('SMA5'),
                lowest_position=self.broker.lowest_point,
                n_long=n_long,
                n_short=n_short,
                max_order=self.broker.im.maxOrder,
                inventory=self.broker.im.inventory,
            )

        if self.observation_provider.type == available_strategy[1]:
            params = ParamsObsDonchian(
                mark_price=self.broker.row_data['SMA5'],
                avg_position=self.broker.im.getAveragePrice('SMA5'),
                lowest_position=self.broker.lowest_point,
                last_buy_position=self.broker.im.getLastEntry('SMA5'),
                inventory=self.broker.im.inventory,
                max_inventory=self.broker.im.maxOrder,
                upper_channel=self.broker.row_data['DC_UPPER'],
                lower_channel=self.broker.row_data['DC_LOWER'],
                upper_channel_shift=self.broker.row_data['DC_UPPER_CHANGES'],
                lower_channel_shift=self.broker.row_data['DC_LOWER_CHANGES'],
                lower_shift_5=self.broker.row_data['DC_LOWER_CHANGES_5_ROW'],
                upper_shift_5=self.broker.row_data['DC_UPPER_CHANGES_5_ROW']
            )

        return self.observation_provider.get_observation(params)

    def check_if_terminated(self):
        if self.terminated:
            priceChange = (self.broker.row_data['SMA5'] /  self.initial_price) * 100
            print(f"An eps end, final balance: {int(self.broker.ac.balance)}, steps: {self.current_step}, trades: {self.broker.im.tradeTotal}, reward: {self.broker.reward_cum:.4f}, price change: {priceChange:.2f}%")
