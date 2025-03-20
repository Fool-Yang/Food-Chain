from env.food_chain import FoodChain

from pettingzoo.test import parallel_api_test


if __name__ == "__main__":
    env = FoodChain()
    parallel_api_test(env, num_cycles=1_000_000)

    env = FoodChain()
    observations, infos = env.reset()
    env.print_game_board()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.print_game_board()
    env.close()
