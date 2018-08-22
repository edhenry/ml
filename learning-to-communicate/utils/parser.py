import argparse

def get_default_parser():
    parser = argparse.ArgumentParser(description="Learning to Communicate with Deep Multi-Agent Reinforcement Learning")
    
    #general options
    parser.add_argument('--seed', type=int, default=42, 
                        help="Initial Random Seed (default: 42)")
    parser.add_argument('--threads', type=int, defualt=1,
                        help='Number of threads (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help="disable cuda and run training on CPU's (default : train using CUDA)")
    
    # reinforcement learning options
    parser.add_argument('--gamma', type=float, default=1,
                        help="Discount factor for RL algorithm(s) (default: 1)")
    parser.add_argument('--eps', type=float, default=0.5,
                        help="Epsilon greedy policy (default: 0.5)")
    
    # Model params
    parser.add_argument('--model', type=str, default="gru",
                        help="Type of hidden unit to use in Recurrent Neural Network (default: GRU)")
    parser.add_argument('--model-dial', type='store_true', default=True,
                        help="Use RIAL (False) or DIAL (True) (default: False - use RIAL)")
    parser.add_argument('--model-comm-narrow', type='store_true', default=True,
                        help="Enable (True) or disable (False) combinning comm bits (default: 1)")
    parser.add_argument('--model-know-share', type='store_true', default=True,
                        help="Enable (True) or disable (False) knowledge sharing")
    parser.add_argument('--model-action-aware', type='store_true', default=True,
                        help="Enable (True) or disable (False) using last action as input to next time step (default: True) ")
    parser.add_argument('--model-rnn-size', type=int, default=128,
                        help="RNN roll out length (default: 128)")
    parser.add_argument('--model-rnn-layers', type=int, default=2,
                        help="Number of layers for depth of RNN (default: 2)")
    parser.add_argument('--model-dropout', type='store_true', default=True,
                        help="Enable (True) or disable (False) dropout for model training (default: True)")
    parser.add_argument('--model-batch-norm', type='store_true', default=True,
                        help-"Enable (True) or disable (False) batch normalization for training of model (default: True)")
    parser.add_argument('--model-target', type='store_true', default=True,
                        help="Enable (True), or disable (False) use of a target network (default: True)")
    parser.add_argument('--model-avg-q', type='store_true', default=True,
                        help="Enable (True) or disable (False) averaging of q functions (default: True)")
    
    # training settings
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help="Learning rate for model training (default: 0.0001)")
    parser.add_argument('--num-episodes', type=float, default=1e+6,
                        help='Number of episodes (default 1e+6')
    parser.add_argument('--num-steps', type=int, default=10,
                        help="Number of steps (default: 10)")
    parser.add_argument('--step', type=int, default=10,
                        help="Print every n steps (default: 10)")
    parser.add_argument('--step-test', type=int, default=10,
                        help="Print every n test steps (default: 10)")
    #TODO Implement the filename arg for checkingpointing model while training
    parser.add_argument('--filename', type=str, help="Filename for use in checkpointing model(s)")
    
    
    # game settings
    #TODO implement color digits game
    # ColorDigits

    # Switch
    parser.add_argument('--game', type=str, default='switch',
                        help="Which game the agents should try playing (default: switch)")
    parser.add_argument('--game-num-agents', type=int, default=3,
                        help="Number of agents to use while playing a game (default: 3)")
    parser.add_argument('--game-action-space', type=int, )
    parser.add_argument('--game-comm-limited', type='store_true', default=True,
                        help="Enable or disable limited comms between agents for the game (default: True)")
    parser.add_argument('--game-comm-bits', type=int, default=2,
                        help="Number of bits used for communications between agents (default: 2)")
    #TODO update with explanation of the sigma hyperparameter for the switch game
    parser.add_argument('--game-comm-sigma', type=int, default=0,
                        help="Hyperparameter")
    parser.add_argument('--nsteps', type=int, default=6,
                        help="Number of steps to allow game to execute")
    