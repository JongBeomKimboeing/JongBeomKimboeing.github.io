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
lower() / upper()는 원래 문열은 수정하지 않는다.]

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

<br>
<br>
<br>
<br>

# 딕셔너리
#### 기본적인 딕셔너리 활용

<pre>
{key: value}
 -> key: 값을 찾기 위해 넣어주는 데이터, value: 찾고자하는 데이터
 원하는 데이터를 빠르게 찾기 위해 사용한다.
</pre>

ex)
txt파일에서 데이터를 가져와 dictionary 만들어주기

```python
source_file = "netflix.txt"

def make_dictionary(filename):
    user_to_titles = {}
    with open(filename) as file:
        for line in file:
            user, title = line.strip().split(':')
            user_to_titles[user] = title

        return user_to_titles

# 아래 주석을 해제하고 결과를 확인해보세요.
print(make_dictionary(source_file))
```

#### 딕셔너리 키
딕셔너리의 키는 변화할 수 없는 값만 가능하다.<br>
그러므로, 딕셔너리 키를 두 개 이상으로 줄 경우 튜플로 묶어준다.<br>

ex)<br>
딕셔너리 키 확인하기

```python
account = {"kdhong":"Kildong Hong",}
print("kdhong" in account)
print("elice" in account)
```

ex)<br>
딕셔너리 순회하기<br>

.items()는 튜플 형태로 key와 value를 반환해준다. ("kdhong","Kildong Hong")

```python
account = {"kdhong":"Kildong Hong",}
for username, name in account.items():
    # account.items()는 튜플 형태로 key와 value를 반환해준다. ("kdhong","Kildong Hong")
    print(username + '-' + name)
```

ex)<br>
사용자가 시청한 작품의 리스트를 저장하고 개수를 샌다.

```python
user_to_titles = {
    1: [271, 318, 491],
    2: [318, 19, 2980, 475],
    3: [475],
    4: [271, 318, 491, 2980, 19, 318, 475],
    5: [882, 91, 2980, 557, 35],
}


def get_user_to_num_titles(user_to_titles):
    user_to_num_titles = {}
    for data, wlist in user_to_titles.items():
        user_to_num_titles[data] = len(wlist)


    return user_to_num_titles

print(get_user_to_num_titles(user_to_titles))
```

<br>
<br>
<br>
<br>

# Json 파일 다루기
#### Json을 딕셔너리로 바꿀 경우 -> loads() 이용
#### 딕셔너리를 Json으로 바꿀 경우 -> dumps() 이용

```python
# json 패키지를 임포트합니다.
import json


# loads()
# JSON 파일을 읽고 문자열을 딕셔너리로 변환합니다.
#-------------------------------------------------------
def create_dict(filename):
    with open(filename) as file:
        json_string = file.read()
        dict = json.loads(json_string)
        # 함수를 완성하세요.
        return dict
#-------------------------------------------------------


# dumps()
# JSON 파일을 읽고 딕셔너리를 JSON 형태의 문자열로 변환합니다.
#-------------------------------------------------------
def create_json(dictionary, filename):
    with open(filename, 'w') as file:
        # 함수를 완성하세요.
        jsonf = json.dumps(dictionary)
        file.write(jsonf) # 파일에 수정한 dictionary를 적어줘야함
        pass
#-------------------------------------------------------


src = 'netflix.json'
dst = 'new_netflix.json'

netflix_dict = create_dict(src)
print('원래 데이터: ' + str(netflix_dict))

# 생성된 dictionary에 원소 추가
netflix_dict['Dark Knight'] = 39217
# dictionary를 json으로 변환
create_json(netflix_dict, dst)
updated_dict = create_dict(dst)
print('수정된 데이터: ' + str(updated_dict))
```
<br>
<br>
<br>
<br>

# 집합
집합은 중복이 없고, 순서가 없다.
#### 집합은 key와 value가 없고 ','로 구분한다.

```python
set1 = {1,2,3} # 집합은 key와 value가 없고 ','로 구분한다.
```

#### 리스트를 set으로 변환

```python
set2 = set([1,2,3]) # 리스트를 set으로 변환
```

#### set의 성질

- set([1,2,3])과 set([3,2,1])은 같은 데이터이다. 왜냐하면, 집합은 순서가 상관이 없기 때문이다.

```python
set3 = {3,2,3,1} # -> 집합은 중복이 없기 때문에 {3,2,3,1} 또한 {1,2,3}과 같은 집합으로 본다.
```

#### set의 원소 추가/삭제 (직접 수정한다.)
- add(data) -> 원소 추가
- update([list]) -> list안에 들어있는 데이터 원소들을 set에 넣어준다.
- remove(data) -> 원소 삭제 (반드시 set에 원소가 존재해야함 set에 없는 원소면 error)
- discard(13) -> 원소 삭제 (원소가 있다면 삭제 없다면 무시)

```python
num_set = {1,3,5,7}
num_set.add(9) # 원소 추가
print(num_set) # {1, 3, 5, 7, 9}
num_set.update([3, 15, 4]) # list안에 들어있는 데이터 원소들을 set에 넣어준다.
print(num_set)
num_set.remove(7) # 원소 삭제 (반드시 set에 원소가 존재해야함 set에 없는 원소면 error)
num_set.discard(13) # 원소 삭제 (원소가 있다면 삭제 없다면 무시)
```

#### 집합 다루기
ex)<br>
아래와 같이 in과 len 사용 가능

```python
num_set = {1,3,5,7}
print(6 in num_set)
print(len(num_set))
```

#### 집합의 연산

- 합집합: set1 | set2 
- 교집합: set1 & set2
- 차집합: set1 - set2 
- xor: set1 ^ set2 

```python
set1 = {1,3,5,7}
set2 = {1,3,9,27}

union = set1 | set2 # 합집합
print(union) # {1, 3, 5, 7, 9, 27}

intersection = set1 & set2 # 교집합
print(intersection) # {1, 3}

diff = set1 - set2 # 차집합
print(diff) # {5, 7}

xor = set1 ^ set2 # xor
print(xor) # {5, 7, 9, 27}
```

<br>
<br>
<br>
<br>

# 그래프 다루기
아래 코드를 참고하여 그래프 그리는데 사용한다.

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 날짜 별 온도 데이터를 세팅합니다.
dates = ["1월 {}일".format(day) for day in range(1, 32)]
temperatures = list(range(1, 32))


# 막대 그래프의 막대 위치를 결정하는 pos를 선언합니다.
pos = range(len(dates))


# 한국어를 보기 좋게 표시할 수 있도록 폰트를 설정합니다.
font = fm.FontProperties(fname='./NanumBarunGothic.ttf')


# 막대의 높이가 빈도의 값이 되도록 설정합니다.
plt.bar(pos, temperatures, align='center')


# 각 막대에 해당되는 단어를 입력합니다.
plt.xticks(pos, dates, rotation='vertical', fontproperties=font)
#pos 위치에 dates를 넣어준다.


# 그래프의 제목을 설정합니다.
plt.title('1월 중 기온 변화', fontproperties=font)


# Y축에 설명을 추가합니다.
plt.ylabel('온도', fontproperties=font)


# 단어가 잘리지 않도록 여백을 조정합니다.
plt.tight_layout()


# 그래프를 표시합니다.
plt.show()
```

<br>
<br>
<br>
<br>

# CSV 읽어오기

### 1. CSV 읽어오는 방법

```python
import csv

with open('movies.csv') as file:
    reader = csv.reader(file, delimiter=',') # 파일 읽어오기
    for row in reader:
        print(row[0])
```

### 2. CSV 파일에서 데이터 추출

```python
import csv

def print_book_info(filename):

    with open(filename) as file:
        # ',' 기호로 분리된 CSV 파일을 처리하세요..
        reader = csv.reader(file, delimiter=',')
        # 처리된 파일의 각 줄을 불러옵니다.
        for row in reader:
            # 함수를 완성하세요.
            title = row[0]
            author = row[1]
            pages = row[3]
            print("{} ({}): {}p".format(title, author, pages))


# 아래 주석을 해제하고 실행 결과를 확인해보세요.
filename = 'books.csv'
print_book_info(filename)
```

### 3. CSV 데이터를 JSON 형식으로 저장

- reader = csv.reader(src, delimiter=',') 를 이용하여 데이터를 읽어온다.
- reader를 info를 통해 for문으로 한줄 한줄 가져온다.
- info를 인덱싱하여 딕셔너리로 저장한다.
- with open(dst_file, 'w') as dst: 를 이용하여 dst_file을 쓰기모드로 연다.
- jbook = json.dumps(books) 딕셔너리를 JSON 형식으로 변환한다.
- dst.write(jbook)  dst_file에 쓴다.


```python
import csv
import json

def books_to_json(src_file, dst_file):
    # 아래 함수를 완성하세요.
    books = []
    with open(src_file) as src:
        reader = csv.reader(src, delimiter=',')

        # 각 줄 별로 대응되는 book 딕셔너리를 만듭니다.
        for info in reader:
            # 책 정보를 저장하는 딕셔너리를 생성합니다.
            book = {
                'title': info[0],
                'author': info[1],
                'genre': info[2],
                'pages': int(info[3]),
                'publisher': info[4]
            }
            books.append(book)

    with open(dst_file, 'w') as dst:
        # JSON 형식으로 dst_file에 저장합니다.
        jbook = json.dumps(books)
        dst.write(jbook)
        pass

src_file = 'books.csv'
dst_file = 'books.json'
books_to_json(src_file, dst_file)
```

<br>
<br>
<br>
<br>

# 고급 파이썬

### 1. lambda
- 함수를 간단하게, 짧게 만들 수 있다.

#### 1) lambda 형식

lambda 입력: 리턴값

#### 2) lambda를 잘 만들었는지 확인 방법

assert() -> true면 아무것도 안 함, false이면 error가 뜸 (lambda를 잘 만들었는 지 확인 할 때 쓴다.)

#### lambda와 assert 사용 예시

```python
#num을 제곱한 값을 리턴합니다.

def _square(num):
    return num * num

# _square()와 동일한 기능을 하는 lambda 함수 square를 만들어 보세요.
square = lambda x: x*x

#string이 빈 문자열일 경우 빈 문자열을, 아니면 첫 번째 글자를 리턴합니다.

def _first_letter(string):
    return string[0] if string else ''

first_letter = lambda string: string[0] if string else ''


# assert를 이용하여 두 함수의 기능이 동일한 지 테스트합니다. 아래 주석을 해제하고 결과 값을 확인해보세요.
testcases1 = [3, 10, 7, 1, -5]
for num in testcases1:
    assert(_square(num) == square(num))

testcases2 = ['', 'hello', 'elice', 'abracadabra', '  abcd  ']
for string in testcases2:
    assert(_first_letter(string) == first_letter(string))

# # 위의 assert 테스트를 모두 통과해야만 아래의 print문이 실행됩니다.
print("성공했습니다!")
```

<br>
<br>

### 2. 함수를 리턴하는 함수

- min_validator, max_validator 가 helper 함수를 return 하는 것을 볼 수 있다.
- return 된 helper 함수를 사용하는 것도 주목하자.

```python
def min_validator(minimum):
    def helper(n):# n의 타입이 정수가 아니면 False를 리턴합니다.
        if type(n) is not int:
            return False
        if n < minimum:
            return False
        else:
            return True
    # 아래 함수를 완성하세요.
    return helper


def max_validator(maximum):
    def helper(n):
        # n의 타입이 정수가 아니면 False를 리턴합니다.
        if type(n) is not int:
            return False

        # 아래 함수를 완성하세요.
        if n > maximum:
            return False
        else:
            return True

    return helper


def validate(n, validators):
    # validator 중 하나라도 통과하지 못하면 False를 리턴합니다.
    for validator in validators:
        if not validator(n):
            return False

    return True

# 작성한 함수를 테스트합니다. # 아래 주석을 해제하고 결과 값을 확인해보세요.
# # 나이 데이터를 검증하는 validator를 선언합니다.
age_validators = [min_validator(0), max_validator(120)]
ages = [9, -3, 7, 33, 18, 1999, 287, 0, 13]

# # 주어진 나이 데이터들에 대한 검증 결과를 출력합니다.
print("검증 결과")
for age in ages:
    result = "유효함" if validate(age, age_validators) else "유효하지 않음"
    print("{}세 : {}".format(age, result))
```

<br>
<br>

### 3. map

#### 1) map의 역할
- 어떤 데이터가 주어졌을 때 데이터의 원소에 대해서 동일한 함수를 취해준다.

#### 2) map의 형식과 의미
- map(함수, 리스트) -> 리스트 원소 각각에 함수를 적용해라.

#### 3) map의 이용
<pre>
map의 리턴은 리스트가 아니라, map type이다.
map은 map을 선언할 당시에는 원소를 함수에 적용시키지 않는다.
다만, map으로 만든 원소를 실질적으로 사용할 때 함수에 원소가 적용된다.
그러므로, map 결과물을 얻고 싶다면, 결과물을 list()형변환을 해야함.
</pre>

#### 4) map 사용 예시

- 아래 코드는 lambda 함수인 get_title을 reader 데이터에 적용시켰다.

```python
import csv

def get_titles(books_csv):
    with open(books_csv) as books:
        reader = csv.reader(books, delimiter=',')
        # 함수를 완성하세요.
        get_title = lambda row: row[0]
        titles = map(get_title, reader)

        return list(titles)

# 작성한 코드를 테스트합니다. 주석을 해제하고 실행하세요.
books = 'books.csv'
titles = get_titles(books)
for title in titles:
    print(title)
```

<br>
<br>

### 4. filter

#### 1) filter의 역할
- 모든 원소에 함수를 적용시켜 true가 나온 결과들만 모아줌.

#### 2) filter 형식
- filter(함수, 리스트)

#### 3) filter의 이용
<pre>
filter 또한 map과 같이 바로 연산을 안 해주고 원소를 이용할 때 연산을 해준다.
그러므로, 출력시 리스트로 보고싶다면, list() 형변환을 해주어야한다.
</pre>

#### 4) filter 사용 예시

```python
starts_with_r = lambda w: w.startswith('r')
words = ['real','man','rhythm','dog']
r_words = filter(starts_with_r, words)
print(list(r_words))
```

#### 5) filter 실습
books.csv 파일을 읽어서 페이지 수가 250이 넘는 책들의 제목을 리스트로 리턴하는 get_titles_of_long_books() 함수를 완성하세요.

```python
# CSV 모듈을 임포트합니다.
import csv
def get_titles_of_long_books(books_csv):
    with open(books_csv) as books:
        reader = csv.reader(books, delimiter=',')
        # 함수를 완성하세요.
        is_long = lambda row: int(row[3]) > 250
        get_title = lambda row: row[0]

        long_books = filter(is_long, reader)
        long_book_titles = map(get_title, long_books)

        return list(long_book_titles)

# 작성한 함수를 테스트합니다. 주석을 해제하고 실행하세요.
books  = 'books.csv'
titles = get_titles_of_long_books(books)
for title in titles:
    print(title)
```




















