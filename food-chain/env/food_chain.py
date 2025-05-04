import functools
from random import Random
from copy import deepcopy
from itertools import product

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

# define constants
SPACE = "space"
WALL = "wall"
FOOD = "food"
HOPPER = "hopper"
FROG = "frog"
SNAKE = "snake"
AGENT_TYPES = [HOPPER, FROG, SNAKE]

# define the numeric inputs for MARL algorithms
GAME_ELEMENTS = [SPACE, WALL, FOOD] + AGENT_TYPES
INT_VALUES = {GAME_ELEMENTS[i]: i for i in range(len(GAME_ELEMENTS))}

# define the food chain as a directed graph
PREDATORS = {
    FOOD: set([HOPPER]),
    HOPPER: set([FROG]),
    FROG: set([SNAKE]),
    SNAKE: set([])
}

class FoodChain(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "food_chain_v0",
    }

    def __init__(self, init_board=None, height=8, width=8, population={}, vision_radius=4, food_reward=0.1, max_steps=64):
        """The init method takes in environment arguments.
        """
        self.possible_agents = []

        self.init_board = init_board
        if init_board:
            self.height = len(init_board)
            self.width = len(init_board[0])
            agent_types = set(AGENT_TYPES) # build a set for efficient lookup
            for y in range(self.height):
                for x in range(self.width):
                    element = self.init_board[y][x]
                    element_type = element.split('_')[0]
                    if element_type in agent_types:
                        self.possible_agents.append(element)
        else:
            self.height = height
            self.width = width
            for agent_type in AGENT_TYPES:
                for i in range(population.get(agent_type, 1)): # default agent number = 1
                    agent = agent_type + "_" + str(i)
                    self.possible_agents.append(agent)
            self.number_of_possible_agents = len(self.possible_agents)

        # vision_radius=None means full observability
        if vision_radius is None:
            vision_radius = max((self.height, self.width))
        self.vision_radius = vision_radius
        self.vision_diameter = self.vision_radius*2 + 1

        self.food_reward = food_reward
        self.max_steps = max_steps

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.RNG = Random()
        if seed is not None:
            self.RNG.seed(seed)

        # initialize agents
        self.agents = deepcopy(self.possible_agents)
        self.agent_positions = {agent: None for agent in self.agents}
        self.empty_positions = set(product(range(self.height), range(self.width))) # for spawning things on empty tiles
        # dead agents enter an absorbing state instead of being removed from the game
        # they receive no rewards and no observations
        self.dead_agents = set()

        # draw the game board
        if self.init_board:
            self.grid = deepcopy(self.init_board)
            for y in range(len(self.grid)):
                for x in range(len(self.grid[0])):
                    element = self.grid[y][x]
                    if element != SPACE:
                        self.empty_positions.remove((y, x))
                        if element in self.agent_positions:
                            self.agent_positions[element] = (y, x)
        else:
            self.grid = [[SPACE for _ in range(self.width)] for _ in range(self.height)]

            # divide the map into 4x4 blocks for placing walls and agents
            blocks_in_a_row = self.width//4
            blocks_in_a_column = self.height//4
            blocks = blocks_in_a_row*blocks_in_a_column

            # choose some random blocks for placing agents
            random_block_choices = self.RNG.sample(range(blocks), self.number_of_possible_agents)

            # draw the agents set their positions
            for i in range(self.number_of_possible_agents):
                block_id = random_block_choices[i]
                y, x = block_id//blocks_in_a_row*4, block_id%blocks_in_a_row*4 # translate the block number into coordinates
                self.grid[y][x] = self.agents[i]
                self.agent_positions[self.agents[i]] = (y, x)
                self.empty_positions.remove((y, x))

            # draw walls on the board
            for i in range(blocks):
                y, x = i//blocks_in_a_row*4 + 1, i%blocks_in_a_row*4 + 1 # translate the block number into coordinates
                if self.RNG.random() <= 0.5:
                    self.grid[y][x] = WALL
                    self.empty_positions.remove((y, x))
                if self.RNG.random() <= 0.5:
                    self.grid[y + 1][x] = WALL
                    self.empty_positions.remove((y + 1, x))
                if self.RNG.random() <= 0.5:
                    self.grid[y][x + 1] = WALL
                    self.empty_positions.remove((y, x + 1))
                if self.RNG.random() <= 0.5:
                    self.grid[y + 1][x + 1] = WALL
                    self.empty_positions.remove((y + 1, x + 1))

        observations = {agent: self.observe(agent) for agent in self.agents}

        # get dummy infos (necessary for proper parallel_to_aec conversion)
        infos = {agent: {} for agent in self.agents}

        self.timestep = 0

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        And any internal state used by observe() or render()
        """
        self.timestep += 1

        # move the agents
        # create a random order of executing agents' actions
        agents = list(actions)
        self.RNG.shuffle(agents)
        agents_that_got_food = []
        new_dead_agents = []
        # relaxation algorithm to execute as many actions as possible
        agents_that_want_to_move = set(agents) # create a set of agents that haven't moved yet
        # loop until no more actions can be executed
        agent_moved = True
        while agent_moved:
            agent_moved = False
            for agent in list(agents_that_want_to_move): # make a copy of the set because it will be modified in the for loop
                if agent in self.dead_agents:
                    continue
                y, x = self.agent_positions[agent]
                action = actions[agent]
                # 0, 1, 2, 3, 4 = pass, up, down, left, right
                if action == 0:
                    agents_that_want_to_move.remove(agent)
                    continue
                elif action == 1:
                    new_y = y - 1
                    new_x = x
                elif action == 2:
                    new_y = y + 1
                    new_x = x
                elif action == 3:
                    new_y = y
                    new_x = x - 1
                elif action == 4:
                    new_y = y
                    new_x = x + 1
                if 0 <= new_y and new_y < self.height and 0 <= new_x and new_x < self.width:
                    destination = self.grid[new_y][new_x]
                    if destination == SPACE:
                        self._move_piece(y, x, new_y, new_x)
                        agents_that_want_to_move.remove(agent)
                    elif destination == FOOD:
                        agents_that_got_food.append(agent)
                        self._move_piece(y, x, new_y, new_x)
                        agents_that_want_to_move.remove(agent)
                    elif destination == WALL:
                        agents_that_want_to_move.remove(agent)
                else:
                    agents_that_want_to_move.remove(agent)

        # assign rewards
        rewards = {agent: 0.0 for agent in self.agents}
        for agent in self.agents:
            if agent in self.dead_agents:
                continue
            predators = []
            agent_type = agent.split('_')[0]
            y, x = self.agent_positions[agent]
            up = y - 1
            down = y + 1
            left = x - 1
            right = x + 1
            if up >= 0 and self.grid[up][x].split('_')[0] in PREDATORS[agent_type]:
                predators.append(self.grid[up][x])
            if down < self.height and self.grid[down][x].split('_')[0] in PREDATORS[agent_type]:
                predators.append(self.grid[down][x])
            if left >= 0 and self.grid[y][left].split('_')[0] in PREDATORS[agent_type]:
                predators.append(self.grid[y][left])
            if right < self.width and self.grid[y][right].split('_')[0] in PREDATORS[agent_type]:
                predators.append(self.grid[y][right])
            if predators:
                new_dead_agents.append(agent)
                rewards[agent] += -1.0
                reward = self.food_reward/len(predators)
                for predator in predators:
                    rewards[predator] += reward
        for agent in agents_that_got_food:
            agent_type = agent.split('_')[0]
            if agent_type in PREDATORS[FOOD]:
                rewards[agent] += self.food_reward

        # spawn food
        # each empty tile has a chance to spawn food at each timestep
        for y, x in list(self.empty_positions):
            # it's recommended that the chance is no more than 1%
            if self.RNG.random() <= 0.005:
                self.grid[y][x] = FOOD
                self.empty_positions.remove((y, x))

        # get observations
        observations = {agent: self.observe(agent) for agent in self.agents}

        self.done = self.timestep >= self.max_steps
        # check termination conditions
        terminations = {agent: self.done for agent in self.agents}
        # check truncation conditions (overwrites termination conditions)
        truncations = {agent: self.done for agent in self.agents}

        # get dummy infos
        infos = {a: {} for a in self.agents}

        # remove dead agents
        for agent in new_dead_agents:
            y, x = self.agent_positions[agent]
            self.grid[y][x] = SPACE
            self.empty_positions.add((y, x))
            self.agent_positions[agent] = None
            self.dead_agents.add(agent)
        if self.done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # draw the numerical game board centered around an agent
    def observe(self, agent):
        # return a dummy observation if the agent is dead
        if agent in self.dead_agents:
            return np.array([[INT_VALUES[WALL] for j in range(self.vision_diameter)] for i in range(self.vision_diameter)], dtype=np.float32)
        vision_grid = [[0 for j in range(self.vision_diameter)] for i in range(self.vision_diameter)]
        y, x = self.agent_positions[agent]
        for i in range(len(vision_grid)):
            map_y = y - self.vision_radius + i
            for j in range(len(vision_grid[0])):
                map_x = x - self.vision_radius + j
                if 0 <= map_y and map_y < len(self.grid) and 0 <= map_x and map_x < len(self.grid[0]):
                    vision_grid[i][j] = INT_VALUES[self.grid[map_y][map_x].split('_')[0]]
                else:
                    vision_grid[i][j] = INT_VALUES[WALL]
        return np.array(vision_grid, dtype=np.float32)

    def print_game_board(self):
        printing_symbol = {
            FOOD: '*',
            HOPPER: 'h',
            FROG: 'f',
            SNAKE: 's',
            SPACE: '.',
            WALL: 'X'
        }
        print('\n'.join([' '.join([printing_symbol[cell.split('_')[0]] for cell in row]) for row in self.grid]), end = "\n\n")

    def _move_piece(self, y, x, new_y, new_x):
        self.grid[new_y][new_x] = self.grid[y][x]
        self.grid[y][x] = SPACE
        self.empty_positions.add((y, x))
        try:
            self.empty_positions.remove((new_y, new_x))
        except KeyError:
            pass
        self.agent_positions[self.grid[new_y][new_x]] = (new_y, new_x)

    # observation space should be defined here.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([[len(GAME_ELEMENTS)]*self.vision_diameter]*self.vision_diameter)

    # action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
