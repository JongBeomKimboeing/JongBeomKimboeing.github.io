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
trump_tweets = ['thank', 'you', 'to', 'president', 'moon', 'of', 'south', 'korea', 'for', 'the', 'beautiful',
                'welcoming', 'ceremony', 'it', 'will', 'always', 'be', 'remembered']


def print_korea(text):
    for i in text:
        if(i[0]=='k'):
            print(i)

print_korea(trump_tweets)
```
<br>
<br>

# 문자열 함수
(문자열과 관련된 함수)

### 1. startwith()
- 문자 혹은 문자열로 시작하는 단어를 찾아준다.<br>
ex1)
```python
word = "superman"
print(word.startswith('s'))

# 문자 검사
if word.startswith('a'):
    print("a로 시작하는 단어")
else:
    print("a로 시작하지 않는 단어")

# 문자열도 가능
if word.startswith('super'):
    print("super로 시작하는 단어")
else:
    print("super로 시작하지 않는 단어")
```
ex2)
```python

trump_tweets = ['thank', 'you', 'to', 'president', 'moon', 'of', 'south', 'korea', 'for', 'the', 'beautiful',
                'welcoming', 'ceremony', 'it', 'will', 'always', 'be', 'remembered']


def print_korea(text):
    for i in text:
        if i.startswith('k'):
            print(i)

print_korea(trump_tweets)
```
<br>

### 2. split()
- split -> 어떤 문자열을 기준으로 쪼게어 list로 만들어준다.
```python
intro = "my name is elice"
print(intro.split()) # 공백 기준 쪼개기

alphabet = "a,b,c,d"
print(alphabet.split(',')) # , 기준 쪼개기
```
- split으로 공백 나누기
```python
numbers = "  1  2  3  "
print(numbers.split()) # ['1', '2', '3']  ->  자동으로 공백을 모두 없애준다.
print(numbers.split(' ')) # ['', '', '1', '', '2', '', '3', '', ''] -> 공백을 인식한다.
```
- split으로 모든 개행 문자들을 없애주기.
<pre>
' ' -> 빈칸
'\t' -> tab
'\n' -> Newline

split() 괄호 안에 위 문자를 넣어 개행 문자들을 없애줄 수 있다.
</pre>

```python
trump_tweets = "thank you to president moon of south korea for the beautiful welcoming ceremony it will always be remembered"

def break_into_words(text):
    words= text.split()
    return words
    
print(break_into_words(trump_tweets))
```
<br>
<br>

### 3. append
- list에 원소를 추가한다.<br>
ex)
```python
numers = []
numers.append(1)
print(numers)
numers.append(2)
print(numers)
```
ex)<br>
- 아래 코드는 10보다 작은 수를 원소로 추가하는 코드이다.
```python
numbers = [1,2,10,17]
small_num = []

for num in numbers:
    if num<10:
        small_num.append(num)
print(small_num)
```
ex)<br>
- b로 시작하는 단어를 원소로 추가하는 코드
```python
trump_tweets = ['america', 'is', 'back', 'and', 'we', 'are', 'coming', 'back', 'bigger', 'and', 'better', 'and',
                'stronger', 'than', 'ever', 'before']


def make_new_list(text):

    new_list = []
    for i in text:
        if i.startswith('b'):
            new_list.append(i)
    return new_list


new_list = make_new_list(trump_tweets)
print(new_list)
```
<br>
<br>

### 4. lower() / upper()
#### lower() -> 문자 전체를 소문자로 변경한다.
#### upper() -> 문자 전체를 대문자로 변경한다.
ex)<br>
lower() / upper()는 원래 문열은 수정하지 않는다.
```python
intro = 'My name is Elice'
print(intro.upper()) # 문자 전체를 대문자로 변경 (원래 문자열은 수정하지 않는다.)
print(intro.lower()) # 문자 전체를 소문자로 변경 (원래 문자열은 수정하지 않는다.)
intro = intro.lower() # 값을 직접적으로 변경
print(intro)
```
ex)
```python
trump_tweets = [
    "FAKE NEWS - A TOTAL POLITICAL WITCH HUNT!",
    "Any negative polls are fake news, just like the CNN, ABC, NBC polls in the election.",
    "The Fake News media is officially out of control.",
]


def lowercase_all_characters(text):
    processed_text = []

    for i in text:
        processed_text.append(i.lower())
    return processed_text

print('\n'.join(lowercase_all_characters(trump_tweets)))
```
<br>
<br>

### 5. replace()
- (replace(변경할 문자, 변경 문자))<br>
ex)
```python
intro = "제 이름은 Elice 입니다."
print(intro.replace('Elice','엘리스')) # (원래 문자열은 수정하지 않는다.)
print(intro.replace(' ', '')) # 문자를 없앨수도 있다. (원래 문자열은 수정하지 않는다.)
print(intro)
intro = intro.replace('Elice','엘리스')
print(intro)
```
ex)<br>
- replace를 연속해서 쓸 수 있다.

```python
trump_tweets = [
    "i hope everyone is having a great christmas, then tomorrow it’s back to work in order to make america great again.",
    "7 of 10 americans prefer 'merry christmas' over 'happy holidays'.",
    "merry christmas!!!",
]


def remove_special_characters(text):
    processed_text = []

    for d in text:
        processed_text.append(d.lower().replace('!','').replace(',','').replace("'",''))

    return processed_text

print('\n'.join(remove_special_characters(trump_tweets)))
```
<br>
<br>
<br>
<br>

# 파일 다루기
(파일 읽어오고 활용해는 방식을 다룬다.)<br>

### 1. 파일 읽고 닫기 / 파일 모드 설정
```python
file = open('data.txt') # 파일 열기
cotent = file.read() # 파일 읽어오기   file.write()를 통해 파일을 수정 가능하다.
file.close() # 파일 닫기


# with as 를 이용하면 파일을 자동으로 닫아준다.
with open('data.txt') as file:  # 파일을 file이라는 이름으로 열어오겠다.
    cotent = file.read() # 파일 읽어오기
    # 들여쓰기가 되있는 부분에서만 이 내용이 적용된다.
    # 즉, 들여쓰기가 끝나면 자동으로 파일이 닫힌다.


# 줄 단위로 파일 읽어오기
contents = []
with open('data.txt') as file:
    for line in file:
        contents.append(line)


# 파일의 모드

with open('data.txt', 'w') as file: # w: 쓰기 (write) 모드로 파일을 연다
    file.write('Hello')
```
<br>
<br>

### 2. 파일 내용 한줄 한줄 읽어 출력
ex)
```python
filename = 'corpus.txt'

def print_lines(filename):

    with open(filename) as file:
        line_number = 1

        for data in file:
            print(line_number,data)
            line_number += 1

# print_lines(filename)
```
<br>
<br>
<br>
<br>

# 데이터 구조 다루기 (튜플)
#### 1. 튜플 vs 리스트
<pre>
튜플 vs 리스트
공통점: 순서가 있는 원소들의 집합  -> 인덱싱, 슬라이싱 모두 가능

차이점: 각 원소의 값을 수정할 수 없다.
        원소의 개수를 바꿀 수 없다.
</pre>

- 튜플은 각 원소의 값을 수정할 수 없다.<br>
ex)
```python
hello = ('a','b','c')
hello[0] = 'd'  #error
hello = ('d','b','c') # 이와 같이 다시 저장하는 건 가능하다.
```
ex)<br>
cf) strip(): 문자 앞 뒤에 있는 모든 공백문자를 없애준다.
```python
filename = 'corpus.txt'

def import_as_tuple(filename):
    tuples = []
    with open(filename) as file:
        for line in file:
            tuples.append(tuple(line.strip().split(','))) # strip(): 문자 앞 뒤에 있는 모든 공백문자를 없애준다.

    return tuples

print(import_as_tuple(filename))
```
<br>
<br>
<br>
<br>

# 데이터 구조 다루기 (리스트)
#### 리스트로 리스트 만들기

ex)<br>
각 단어의 첫번쨰 문자를 가져온다.
```python
words = ['life', 'love', 'faith']
first_letters = []
for word in words:
    first_letters.append(word[0])
    # 결과: ['l', 'l', 'f']
print(first_letters)
```
위 코드를 더 간결하게 만들어주면 아래와 같다.<br>
아래와 같이 한 줄로 리스트를 만들어 주는 것을 list comprehension 이라고 한다.
```python
words = ['life', 'love', 'faith']
first_letters = [word[0] for word in words]
print(first_letters)
```

ex)<br>
모든 리스트 원소 하나하나에 1을 더하고 리스트로 만든다.
```python
numbers = [1,3,5,7]
new_numbers = []
for n in numbers:
    new_numbers.append(n+1)
print(new_numbers)
```
위 코드를 더 간결하게 만들어주면 아래와 같다.
```python
numbers = [1,3,5,7]
new_numbers = [n+1 for n in numbers]
print(new_numbers)
```

ex)<br>
모든 리스트 원소 중 짝수 원소를 리스트로 만든다.
```python
numbers = [1,3,4,5,6,7]
even = []
for n in numbers:
    if n % 2 == 0:
        even.append(n)
print(even)
```
위 코드를 더 간결하게 만들어주면 아래와 같다.
```python
numbers = [1,3,4,5,6,7]
even = [n for n in numbers if n % 2 == 0]
print(even)
```

ex)<br>
a로 시작하는 단어 한줄로 추출
```python
words = [
    'apple',
    'banana',
    'alpha',
    'bravo',
    'cherry',
    'charlie',
]

def filter_by_prefix(words, prefix):
    # 아래 코드를 작성하세요.
    wordl = [wor for wor in words if wor.startswith(prefix)]
    return wordl
    
a_words = filter_by_prefix(words, 'a')
print(a_words)
```
<br>
<br>
<br>
<br>

# 데이터 정렬하기
#### sorted를 이용하여 정렬 (sorted는 오름차순이 기본)

ex)
- sorted(numbers, key=abs) -> key에 적용할 함수를 넣어 sort할 조건을 만들어 줄 수 있다.
```python
numbers = [-1,3,-4,5,6,100]
sort_by_abs = sorted(numbers, key=abs) # key에 적용할 함수를 넣어 sort할 조건을 만들어 줄 수 있다.
print(sort_by_abs)
```
- 단어로 이루어진 list를 sorted하면, 사전순으로 정렬해준다.
```python
fruits = ['cherry', 'apple', 'banana']
sort_by_alphabet = sorted(fruits) # 사전순으로 정렬해준다.
print(sort_by_alphabet)
```

#### sorted(list, key=)의 key에 함수를 넣어보기
ex)
- key에 사용자 정의함수를 넣었다.
```python
sort_by_last = []

def reverse(word):
    return str(reversed(word))
    
    
fruits = ['cherry', 'apple', 'banana']
sort_by_last = sorted(fruits, key=reverse) # key에 reverse 함수를 넣었다.
print(sort_by_last)
```
ex)
```python
pairs = [
    ('time', 8),
    ('the', 15),
    ('turbo', 1),
]

#(단어, 빈도수) 쌍으로 이루어진 튜플을 받아, 빈도수를 리턴합니다.
def get_freq(pair):
    return pair[1]


#(단어, 빈도수) 꼴 튜플의 리스트를 받아, 빈도수가 낮은 순서대로 정렬하여 리턴합니다.
def sort_by_frequency(pairs):
    sort = sorted(pairs, key=get_freq)
    return sort


# 아래 주석을 해제하고 결과를 확인해보세요.
print(sort_by_frequency(pairs))
```








































