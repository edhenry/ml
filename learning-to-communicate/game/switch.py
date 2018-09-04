import torch
import logging

class Switch(object):
    """
    Switch game as described in http://arxiv.org/abs/1605.06676

    Implementation modeled after original Lua implementation found at : https://github.com/iassael/learning-to-communicate/blob/master/code/game/Switch.lua

    Arguments:
        object {[type]} -- [description]
    """
    #TODO Define game_reward_shift type
    def __init__(self, game: str, game_num_agents: int, game_action_space: int,
                game_comm_limited: bool, game_comm_bits: int, 
                game_comm_sigma: int, nsteps: int, game_reward_shift,
                batch_size: int, step_counter: int):
        
        self.game = game
        self.game_num_agents = game_num_agents
        self.game_action_space = game_action_space
        self.game_comm_limited = game_comm_limited
        self.game_comm_bits = game_comm_bits
        self.game_comm_sigma = game_comm_sigma
        self.nsteps = nsteps
        self.game_reward_shift = game_reward_shift
        self.batch_size = batch_size

        # Max steps override 
        # Page 6 of http://arxiv.org/abs/1605.06676
        nsteps = 4 * game_num_agents - 6

        self.reward = torch.zeros(self.batch_size, self.game_num_agents)
        self.has_been = torch.zeros(self.batch_size, self.nsteps, self.game_num_agents)
        self.terminal = torch.zeros(self.batch_size)

        self.active_agent

    def reset(self):
        """
        Reset state of switch game
        """
        # reset rewards
        self.reward = torch.zeros(self.batch_size, self.game_num_agents)

        self.has_been = torch.zeros(self.batch_size, self.nsteps, self.game_num_agents)

        self.terminal = torch.zeros(self.batch_size)

        self.step_counter = 1

        self.active