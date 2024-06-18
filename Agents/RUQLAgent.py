#!/usr/bin/env python3
import os
import pickle
import datetime
from os.path import expanduser
import diambra.arena
import numpy as np
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent, EnvironmentSettings, WrappersSettings, RecordingSettings, Roles
from collections import defaultdict

def create_epsilon_greedy_policy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):

        action_probabilities = np.ones(num_actions,
                dtype = float) * epsilon / num_actions

        best_action = np.argmin(Q[state])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policyFunction

# Bucket health into 3 values
def convert_health(health):
    if health > 100:
        return 2
    elif health > 50:
        return 1
    else:
        return 0

# Extract relevant pieces from observation in order to reduce state space
def convert_obs(obs):
    return (convert_health(obs["P1"]["health"]), obs["P1"]["side"], obs["P2"]["character"], obs["P2"]["side"])

def main():
    settings = EnvironmentSettings()

    wrappers_settings = WrappersSettings()
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 2

    recording_settings = RecordingSettings()

    recording_settings.username = "gugam"
    home_dir = expanduser("~")
    game_id = "umk3"
    game_file_path = f'./PickledQTables/{game_id}.pkl'
    recording_settings = RecordingSettings()
    recording_settings.dataset_path = os.path.join(home_dir, "Downloads/DIAMBRAAgents/episode_recording", game_id)


    settings.frame_shape = (128, 128, 0)
    settings.step_ratio = 1
    # Player role selection: P1 (left), P2 (right), None (50% P1, 50% P2)
    settings.role = Roles.P1

    # Game continue logic (0.0 by default):
    # - [0.0, 1.0]: probability of continuing game at game over
    # - int((-inf, -1.0]): number of continues at game over
    #                      before episode to be considered done
    settings.continue_game = 0.99

    # If to show game final when game is completed
    settings.show_final = False

    # Game-specific options (see documentation for details)
    # Game difficulty level
    settings.difficulty = 1

    settings.action_space = SpaceTypes.DISCRETE

    settings.characters = "Skorpion"

    # Environment creation
    env = diambra.arena.make(game_id, settings, episode_recording_settings=recording_settings, wrappers_settings=wrappers_settings, render_mode="human")

    # Environment reset
    obs, info = env.reset(seed=42)
    env.unwrapped.show_obs(obs)
    obs = convert_obs(obs)

    num_actions = env.action_space.n

    #Hyperparameters
    epsilon = 1
    alpha = 0.01
    discount_factor = 0.9

    num_identical_rewards = 0

    try:
        with open(game_file_path, 'rb+') as fp:
            Q = pickle.load(fp)
    except:
        Q = defaultdict(lambda: np.zeros(num_actions))
    if len(Q.keys()) != 0:
        new_Q = defaultdict(lambda: np.zeros(num_actions))
        for key in Q.keys():
            new_Q[key] = Q[key]
        Q = new_Q

    prev_reward = 0
    episode_num = 0
    # Agent-Environment interaction loop
    while True:
        # Environment rendering
        env.render()

        # Decrease epsilon over time
        if episode_num % 1000 == 0:
            epsilon /= 2
        # Use epsilon greedy policy to get action
        policy = create_epsilon_greedy_policy(Q, epsilon, num_actions)
        action_probs = policy(obs)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

        # Environment stepping
        next_obs, reward, terminated, truncated, info = env.step(action)
        if prev_reward != 0 and prev_reward == reward:
            num_identical_rewards += 1
        elif prev_reward != 0 and prev_reward != reward:
            num_identical_rewards = 0
        
        if num_identical_rewards > 5:
            reward *= -1000
        reward *= next_obs['timer']/100


        # Negate reward to use argmin
        reward = -1*reward
        done = terminated or truncated
        next_obs = convert_obs(next_obs)

        # RUQL https://jmlr.org/papers/v17/14-037.html#:~:text=Here%2C%20we%20introduce%20Repeated%20Update,%2Dthe%2Dart%20algorithms%20theoretically.
        left_term = (1-alpha)**(1/action_probs[action])*Q[obs][action]
        right_term = 1 - (1-alpha)**(1/action_probs[action])
        best_next_action = np.argmin(Q[next_obs])
        td_target = reward + discount_factor * Q[next_obs][best_next_action]
        right_term *= td_target
        Q[obs][action] = left_term + right_term

        # Episode end (Done condition) check
        if done:
            obs, info = env.reset()
            break
        
        obs = next_obs
        episode_num += 1
    # Environment shutdown
    env.close()

    print("Q VALUES:")
    for key in Q.keys():
        print(f'KEY: {key}         VALUE: {Q[key]}')

    if not os.path.exists('./PickledQTables/'):
        os.makedirs('./PickledQTables/')
    with open(game_file_path, 'wb+') as fp:
        pickle.dump(dict(Q), fp)
    # Return success
    return 0

if __name__ == '__main__':
    main()