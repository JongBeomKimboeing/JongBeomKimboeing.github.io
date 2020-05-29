---
layout: post
title: Q-learning in Nondeterministic world
description: "Q-learning in Nondeterministic world)"
modified: 2020-05-15
tags: [김성훈, Reinforce Learning]
categories: [김성훈RL]
---
# deterministic한 환경에서 잘 작동했던 코드를 Nondeterministic한 환경에서 적용해보기
## Nondeterministic한 환경의 특징
앞의 환경은 determistic한 환경이었던 반면,<br>
이번 환경은Nondeterministic한 환경이다.<br>
Nondeterministic한 환경이란,<br>
내가 어떤 action을 해도 내 마음대로 action에따른 state를 받지 못하는 환경이다.<br>
아래 코드는 deterministic한 환경에서 잘 작동했던 코드를 Nondeterministic한 환경에서 적용해 본 코드이다.<br>
(환경만 'FrozenLake-v0'로 설정하여 nondeterministic한 환경을 만들어주고 나머지 코드는 전과 동일하다.)

```python
#3
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

env = gym.make('FrozenLake-v0') # 빙판길이 미끄럽게 설정해 주어 nondeterministic한 환경을 만들어준다.

Q = np.zeros([env.observation_space.n, env.action_space.n])

dis = 0.99 # discount factor
num_episode = 2000

rlist = []

for i in range(num_episode):
    state = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i//100)+1)

    while not done:

        if(np.random.rand(1)<e): # e-greedy 기법을 사용했다.
            # 0과1사의의 값을 뽑아 그 값이 e보다 작을 경우
           action = env.action_space.sample() # random하게 action을 취한다.
        else: #그렇지 않은 경우에
            action = np.argmax(Q[state, :])
            # Q-table에서 내가 있는 state에서 취할 수 있는 action들 중 가장 큰 점수의 action을 취한다.

        action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)/(i+1))
        # Q-table에 있는 현재 state에서의 action의 점수와 (1,4)의 shape으로 random하게 숫자를 뽑아 noise를 만들어
        #원래의 action점수와 noise를 합하여 agent가 더 다양한 길을 선택할 수 있도록 도와준다.

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + dis * np.max(Q[new_state,:])
        # Q를 업데이트 시키는데, 현재 받을 reward는 그대로 받고, 미래에 받을 reward를 discount시켜 받는다.
        #-> 더 빠른 길을 찾도록 도와준다.

        rAll += reward
        state = new_state
    rlist.append(rAll)

print("success rate: " + str(sum(rlist)/num_episode))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rlist)), rlist, color="blue")
plt.show()

```
# 실습 결과
성공한 횟수가 아주 적다.
![image](/assets/Q-learning_result.png)
