import random

import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


def make_actor():
    ### Actor Network
    actornet = Sequential()
    actornet.add(Dense(48, activation='relu', input_shape=(2,)))
    # actornet.add(Dropout(0.2))
    # actornet.add(Flatten())
    # actornet.add(Dense(10, activation='relu'))
    # actornet.add(Dropout(0.2))
    # actornet.add(Flatten())
    actornet.add(Dense(3, activation='linear'))
    return actornet


def make_critic():
    ### Critic Network
    criticnet = Sequential()
    criticnet.add(Dense(48, activation='relu', input_shape=(2,)))
    # criticnet.add(Dropout(0.2))
    # criticnet.add(Flatten())
    # criticnet.add(Dense(10, activation='relu'))
    # criticnet.add(Dropout(0.2))
    # criticnet.add(Flatten())
    criticnet.add(Dense(3, activation='linear'))
    return criticnet


def epsilon_greedy(q, st):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * st / eps_decay_steps)
    print(epsilon)
    if (np.random.rand()) <= epsilon and flag:
        return np.random.randint(0, env.action_space.n, size=1)[0]
    else:
        print("aa")
        return np.argmax(q)
    # return np.argmax(q)


def get_y(q_vals, rew):
    act = np.argmax(current_state_pred, axis=1)
    for i in range(q_vals.shape[0]):
        q_vals[i, act[i]] = (1 - learning_rate) * q_vals[i, act[i]] + learning_rate * rew[i]
    return q_vals


def discount_rew(rew):
    l = [0] * (len(rew) + 1)
    for i in range(len(rew) - 1, -1, -1):
        l[i] = l[i + 1] + (rew[i] * discount_rate)
    return l


def update_Q_Table(q, cq):
    act = np.argmax(replay_memory_state, axis=1)
    for i in range(0, len(q)):
        cq[i, act[i]] = ((1 - learning_rate) * cq[i, act[i]]) + learning_rate * (
                replay_memory_reward[i] + (discount_rate * np.max(q[i]) * replay_memory_continues[i]))
    return cq


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    obs = env.reset()
    print(np.array(obs.reshape(-1, 1)).shape)
    print(env.action_space.n)
    actor = make_actor()
    actor.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    critic = make_critic()
    critic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    # actor.load_weights(filepath="actor_weightmountaincar_6.h5")

    replay_memory_state = []
    replay_memory_next_state = []
    replay_memory_action = []
    replay_memory_reward = []
    replay_memory_continues = []

    eps_min = 0.01
    eps_max = 1.0
    eps_decay_steps = 25000
    learning_rate = 0.1
    flag=True

    n_steps = 50000
    step = 0
    iteration = 0
    train_start = 1000
    discount_rate = 0.998
    done = True

    while True:
        env.render()
        step += 1
        print("STEP", step)
        if step > n_steps:
            break
        iteration += 1
        if done:
            obs = env.reset()
            state = obs
        q_values = actor.predict(state.reshape(-1, 2), verbose=0)
        action = epsilon_greedy(q_values, step)
        print(action, obs)
        obs, reward, done, _ = env.step(action)
        if done and obs[0] <= 0.5:
            reward = -2
        if obs[1] <= 0.001 or obs[1] >= -0.001:  # -0.42 >= obs[0] >= -0.58) or
            reward = -2
        if obs[0] >= 0.3:
            flag=False
            reward = 2
        # if obs[0] >= -0.1:
        #     reward = 1
        next_state = obs
        replay_memory_state.append(state)
        replay_memory_next_state.append(next_state)
        replay_memory_action.append(action)
        replay_memory_reward.append(reward)
        replay_memory_continues.append(1 - done)
        state = next_state
        if iteration % train_start == 0:
            print("Start Fitting")
            # print(np.argmax(replay_memory_state))
            # critic.load_weights("critic_weightmountaincar_4.h5")
            next_state_pred = actor.predict(np.array(replay_memory_next_state).reshape(-1, 2), verbose=0)
            current_state_pred = actor.predict(np.array(replay_memory_state).reshape(-1, 2), verbose=0)
            # dis_rew=discount_rew(replay_memory_reward)[0:len(replay_memory_reward)]
            reward_calc = np.array(replay_memory_reward) + discount_rate * np.max(next_state_pred, axis=1) * np.array(
                replay_memory_continues)
            # Q_up = update_Q_Table(next_state_pred, current_state_pred)
            y = get_y(current_state_pred, reward_calc)
            critic.train_on_batch(np.array(replay_memory_state), y)
            actor.set_weights(critic.get_weights())
            # critic.save_weights(filepath="critic_weightmountaincar.h5")
            replay_memory_state = []
            replay_memory_next_state = []
            replay_memory_action = []
            replay_memory_reward = []
            replay_memory_continues = []
            done = True

    actor.save_weights(filepath="actor_weightmountaincar_7.h5")
