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
obs=obs[0]
reward_list=[0]
while not done:
    # env.render()
    step+=1
    pred=actornet.predict(obs.reshape(-1,2))
    action=np.argmax(pred)
    print(action)
    # print(env.step(action))
    obs,reward,done,_,_=env.step(action)
    # print(obs.shape)
    reward_list.append(abs(reward)+reward_list[step-1])
print("Ended at step",step)
print("The total reward",sum(reward_list))
plt.plot(list(range(1,step+1)), reward_list[1:], label='Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Reward over Time (Steps)')
plt.legend()
plt.grid(True)
plt.show()
