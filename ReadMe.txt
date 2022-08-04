******************PROJECT INFO******************
MOUNTAIN CAR DEEP Q LEARNING

PERFORMING DEEP Q LEARNING FOR THE OPENAI'S MOUNTAIN CAR

1) ENVIRONMENT
OpenAi's Mountain Car Environment for training and testing.

2) ACTION SPACE
0,1,2- Three actions for acceleration to left, no acceleration and acceleration to right.

3) OBSERVATIONS SPACE
The sin like road/mountain where the car rolls.

4) POLICY
Deep Q Learning.

5) REWARDS
-1 for each time step.
* But as the car was being stuck in the valley, so I added additional rewards.

        if done and obs[0] <= 0.5:
            reward = -2
        if obs[1] <= 0.001 or obs[1] >= -0.001:  # -0.42 >= obs[0] >= -0.58) or
            reward = -2
        if obs[0] >= 0.3:
            flag=False
	    reward=2

* Where the obs[0] is for location and obs[1] is for velocity.
* The actor_weightmountaincar_5,6,7 are the different configurations that I tested and 7 is the one that reaches the target before 
200 steps.

(A bit of task for you is to find out the use of the variable for flag and what is it flagging at.)

	      ***
