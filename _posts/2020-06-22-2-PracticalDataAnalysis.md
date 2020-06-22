---
layout: post
title: 실전 데이터 분석
description: "파이썬을 활용한 데이터분석"
modified: 2020-06-22
tags: [Data Analysis]
categories: [Data Analysis]
---
# 리스트 순회하기
(for문을 활용하여 리스트를 순회한다.)
```python
trump_tweets = [
    'Will be leaving Florida for Washington (D.C.) today at 4:00 P.M. Much work to be done, but it will be a great New Year!',
    'Companies are giving big bonuses to their workers because of the Tax Cut Bill. Really great!',
    'MAKE AMERICA GREAT AGAIN!'
]
for sen in range(len(trump_tweets)):
    print("2018년 1월 "+str(sen+1)+'일: '+trump_tweets[sen])
```
<br>

# 문자열 인덱싱
(인덱싱을 이용하여 특정 단어를 가져온다.)<br>
아래 예제는 첫문자가 k인 단어를 추출해내는 코드이다.
```python
# 트럼프 대통령 트윗을 공백 기준으로 분리한 리스트입니다. 수정하지 마세요.
trump_tweets = ['thank', 'you', 'to', 'president', 'moon', 'of', 'south', 'korea', 'for', 'the', 'beautiful',
                'welcoming', 'ceremony', 'it', 'will', 'always', 'be', 'remembered']


def print_korea(text):
    for i in text:
        if(i[0]=='k'):
            print(i)
# 아래 코드를 작성하세요.

# 아래 주석을 해제하고 결과를 확인해보세요.
print_korea(trump_tweets)
```


























































