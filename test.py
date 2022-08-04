import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


actornet = Sequential()
actornet.add(Dense(48, activation='relu', input_shape=(2,)))
#actornet.add(Dense(10, activation='relu'))
actornet.add(Dense(3, activation='linear'))

actornet.load_weights('actor_weightmountaincar_7.h5')

step=0
done=False
env = gym.make("MountainCar-v0")
env._max_episode_steps=200
obs = env.reset()
reward_list=[]
while not done:
    env.render()
    step+=1
    pred=actornet.predict(obs.reshape(-1,2))
    action=np.argmax(pred)
    print(action)
    obs,reward,done,_=env.step(action)
    reward_list.append(reward)
print("Ended at step",step)
print("The total reward",sum(reward_list))