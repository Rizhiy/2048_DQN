import random
import math
import torch
import numpy as np

from time import sleep
from DQN import DQN, ReplayMemory, Transition
from itertools import count

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from twenty_forty_eight_python.puzzle import GameGrid, TheGame, Moves
from twenty_forty_eight_python.logic import game_state

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

KEYS = ["'w'", "'s'", "'a'", "'d'"]

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


episode_durations = []

last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


moves = [Moves.UP, Moves.DOWN, Moves.LEFT, Moves.RIGHT]

if __name__ == '__main__':
    num_episodes = 10
    game = TheGame()
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = game.matrix
        for t in count():
            # Select and perform an action
            move = select_action(game.matrix)
            prev_state = game
            done = game.make_move(move)
            current_state = game
            reward = prev_state.score - current_state.score
            reward = Tensor([reward])

            if not done:
                next_state = current_state.matrix
            else:
                next_state = None

            # Store the transition in memory
            memory.push(prev_state.matrix, move, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                break

    grid = GameGrid(game)
    grid.mainloop()
