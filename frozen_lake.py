import math
import numpy as np
import tqdm
import random
import gym
import time
from random_gen import MersenneTwister


class FrozenLake:
    """
    Class for reinforcement learning in game Frozen Lake from Gym library
    """

    def __init__(self,
                 env: gym.wrappers.time_limit.TimeLimit,
                 episodes: int,
                 epsilon: int,
                 gamma: int) -> None:
        """
        Inital setup of parameters
        """
        self.policy = {}
        self.Q = {}

        self.generator = MersenneTwister()

        # Input variables for learning
        self.num_episodes = episodes
        self.hole_counter = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.score = 0

        # Number of places on the board of Frozen Lake game -- 16 or 64
        self.num_places = self.env.observation_space.n

        # Number of actions
        # left - 0, down - 1, right - 2, up - 3
        self.num_actions = self.env.action_space.n

        # Create initial random policy
        self.create_policy()
        # Create initial
        self.create_Q()

    def get_neighbours_count(self, place: int) -> int:
        """
        Function, that returns number of possible directions
        :param place: place, where agent is
        :return: Number of possible directions as int
        """
        sqrt_place = math.sqrt(self.num_places)
        if (place == 0
                or place == int(self.num_places - sqrt_place)
                or place + 1 == int(sqrt_place)
                or place + 1 == self.num_places):
            return 2

        elif (place % sqrt_place == 0):
            return 3

        elif ((place + 1) % 4 == 0):
            return 3

        elif (place >= self.num_places - sqrt_place):
            return 3

        elif (place < sqrt_place):
            return 3

        else:
            return 4

    def create_policy(self) -> None:
        """
        Create random policy for first iteration
        """
        for key in range(self.num_places):
            p = {}
            count = self.get_neighbours_count(key)
            for action in range(self.num_actions):
                p[action] = round(1 / count, 2)
                self.policy[key] = p

    def create_Q(self) -> None:
        """
        Create initial Q matrix for first iteration
        """
        for key in self.policy.keys():
            self.Q[key] = {x: 0.0 for x in range(self.num_actions)}

    def get_episode(self) -> list:
        """
        Run whole game and record every move
        :return: List of timesteps for every move in one episode run
        """
        self.env.reset()
        episode = []
        finished = False
        while not finished:
            pos = self.env.env.s
            upper_boundary = 0
            # for every prob. of actual position in policy
            # prob -- dict_items([(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)])
            for prob in self.policy[pos].items():
                # prob for actual action on position
                upper_boundary += prob[1]
                # print(prob)
                self.generator.set_seed(seed=int(time.time() * 1000))
                if self.generator.get_uniform(0, sum(self.policy[pos].values())) < upper_boundary:
                    action = prob[0]
                    break

            state, reward, finished, _, _ = self.env.step(action)

            episode.append([pos, action, reward])

            if finished and reward == 0: self.hole_counter += 1

        return episode

    def train(self) -> None:
        """
        Reinforcement learning based on Monte Carlo method for policy iteration.
        """
        values = {}
        wins = 0

        # for x episodes/runs
        for _ in range(self.num_episodes):
            # episode reward
            G = 0

            # get one episode
            # state, action, reward
            episode = self.get_episode()

            # If episode was success(agent ended in G) -> wins++
            if episode[-1][-1] == 1: wins += 1

            # reversed due to back propagation
            for i in reversed(range(len(episode))):
                # s_t - state in time t
                # a_t - action in time t
                # r_t - reward in time t
                s_t, a_t, r_t = episode[i]
                state_action = (s_t, a_t)

                # episode return
                G = r_t + self.gamma * G

                if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                    # values item is empty/[]
                    # workaround due to non exist key
                    # {(13,1):[]}
                    if values.get(state_action):
                        values[state_action].append(G)
                    # {(13,1): [0.24]}
                    else:
                        values[state_action] = [G]

                    # nested dictionary {state: {action: q value}}
                    # Avg. reward across all plays
                    self.Q[s_t][a_t] = sum(values[state_action]) / len(values[state_action])

                    # find maximum reward/argmax based on action, get prob of action
                    Q_list = list(map(lambda x: x[1], self.Q[s_t].items()))
                    # indices with maximum reward
                    indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]

                    # new policy creation
                    for a in self.policy[s_t].items():
                        if a[0] == random.choice(indices):
                            # new prob.
                            self.policy[s_t][a[0]] = 1 - self.epsilon + (
                                    self.epsilon / abs(sum(self.policy[s_t].values())))
                        else:
                            self.policy[s_t][a[0]] = (self.epsilon / abs(sum(self.policy[s_t].values())))

        # Calculation of score
        self.score = wins / self.num_episodes

    # Function to test policy and print win percentage

    ## deprecated
    def pseudo_gen(self, mult=16807,
                   mod=(2 ** 31) - 1,
                   seed=12345,
                   size=1) -> int:
        U = np.zeros(size)
        x = (seed * mult + 1) % mod
        U[0] = x / mod
        for i in range(1, size):
            x = (x * mult + 1) % mod
            U[i] = x / mod
        return U

    def uniform_gen(self, low=0, high=1, seed=12345, size=10000) -> float:
        return random.choice(low + (high - low) * self.pseudo_gen(seed=seed, size=size))
