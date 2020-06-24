---
layout: post
title: Dummy Q-Learning
description: "Dummy Q-Learning"
modified: 2020-05-15
tags: [김성훈,Reinforce Learning]
categories: [김성훈RL]
---
# Dummy Q-Leraning
불완전한 Q=Learning코드로 Q-table을 이용하여 agent가 target에도달한다.<br>
진짜 Q-Learning과 달리, exploit and exploration을 사용하지 않고<br>
그냥 우연히 target에 도달해서 Q-table을 업데이트 시키는 방식으로 작동한다.<br>
target에 효율적으로 도달하는 방법을 찾기에는 부족하고, target에 도달하는 방법만 알아낸다.<br> 
```python
#1
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector): # 테이블이 모두 0일 때 random하게 가라.
    m = np.amax(vector) # amax: array의 최댓값을 반환하는 함수
    indices = np.nonzero(vector == m)[0] # nonzero: 요소들 중 0이 아닌 값들의 index를 반환해준다.
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point= 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs= {'map_name': '4x4',
             'is_slippery': False}
)
env = gym.make('FrozenLake-v3') # environment 만들어주기
Q = np.zeros([env.observation_space.n, env.action_space.n]) #16x4의 Q테이블을 만들고 테이블을 모두 0으로 초기화

num_episodes = 2000 # episode 수
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = rargmax(Q[state,:]) # achtion을 준다

        new_state, reward, done, _ = env.step(action) # action에 대한 각각의 값을 받아준다
        Q[state, action] = reward + np.max(Q[new_state, :]) # Q learning의 핵심적인 식

        rAll +=reward # 만약에 goal에 들어갈 경우에만 reward가 1이되고 나머지의 경우는 0
        state = new_state # state를 행동에 의한 새로운 state로 옮겨준다.

    rList.append(rAll)

print("Sucess rate:" + str(sum((rList))/num_episodes)) #정확도 측정
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue") #성공횟수 그래프 그리기
plt.show()
```
