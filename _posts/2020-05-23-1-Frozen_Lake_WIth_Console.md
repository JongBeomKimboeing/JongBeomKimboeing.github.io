---
layout: post
title: Custom Image(내가 가지고 있는 데이터)를  사용하여 Image 분류하기
description: "Image classification with custom image(my data)"
modified: 2020-05-15
tags: [김성훈, Reinforce Learning]
categories: [김성훈RL]
---
# FrozenLake 체험하기
OpenAI gym에 있는 environment 중 하나인 FrozenLake를 체험해보자.<br>
반드시, 실행은 cmd에서 해야하며,<br>
실습목적은 agent가 action을 하면 어떤 결과값을 출력하는 지 알기 위함이다.
```python
#반드시 cmd에서 pycharmproject로 들어가 이 파일을 실행한다.
#pycharm에서 방향키 입력을 못 받아서 cmd로 실행시켜야한다.
import tensorflow as tf
import gym
from gym.envs.registration import register
import msvcrt

import readchar  # pip3 install readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT}

#환경을 등록해준다.
register(
    id='FrozenLake-v3',
    entry_point= 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4','is_slippery': False}
)
#환경을 만들어준다.
env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = readchar.readkey() #키를 받는다.
    if key not in arrow_keys.keys():
        print("Game aborted")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action) #받은 키의 action에 대한 결과값을 인수로 받아온다.
    env.render() #키를 적용한 env를 나타내줌
    print("state:", state, "Action:",action,"Reward:",reward,"Info:",info) #결과값의 출력

    if done: #게임이 끝날 시
        print("Finished with reward", reward)
        break
```
