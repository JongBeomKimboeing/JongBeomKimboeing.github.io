---
layout: post
title: Data Crolling
description: "Data Crolling"
modified: 2020-06-25
tags: [Data Crolling]
categories: [Data Crolling]
---

# 1. 크롤링이란?
**웹페이지에서 필요한 데이터를 추출해내는 작업**
- 크롤링하는 프로그램: 크롤러
- 크롤링은 웹페이지의 정보를 HTML문서로 표현한다.

<br>
<br>

# 2. 크롤링을 위해 필요한 것
웹페이지의 HTML을 얻기 위해 request 라이브러리를 이용하고<br>
가져온 HTML을 분석하기 위해 BeutifulSoup 라이브러리를 사용한다.


<br>
<br>

# 3. Beautifulsoup
<pre><font size="3px">
Beautifulsoup: HTML. XML, JSON등 파일의 구문을 분석하는 모듈
               주로 웹페이지를 표현하는 HTML을 분석하기 위해 사용됨.
</font></pre>

<br>

## 1) HTML파일로 BeautifulSoup 객체를 만들기
- HTML파일로 BeautifulSoup 객체를 만들 수 있다.
- 변수 이름은 관습적으로 soup 라고 짓는다.

```python
from bs4 import BeautifulSoup

#HTML파일로 BeautifulSoup 객체를 만들 수 있다.
#변수 이름은 관습적으로 soup 라고 짓는다.

soup = BeautifulSoup(open('index.html'), "html.parser")
```
BeautifulSoup(open('index.html'), "html.parser")에서<br>
"html.parser"의 의미는, BeautifulSoup 객체에게 "HTML을 분석해라"라고 알려주는 의미이다.

<br>

## 2) find(''), find_all('')
find, find_all 메소드를 이용하여 HTML 태그를 추출할 수 있다.

<br>

- find는 추출한 HTML 태그 하나를 찾는다.
- find_all은 HTML태그를 여러 개 담고 있는 리스트를 얻는다.

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(open('index.html'), "html.parser")

soup.find("p") # html 파일에 처음 등장하는 'p'태그 찾기

soup.find_all("p") #html 파일에 모든 'p' 태그 찾기
```

<br>

- 만약 div 태그 중, 특정 class를 추출하려면?

```python

from bs4 import BeautifulSoup

soup = BeautifulSoup(open('index.html'), "html.parser")

soup.find_all("div", class_ = "elice") #class_ 매개변수에 값을 저장함으로써 특정 클래스를 가진 태그를 추출할 수 있다.

```

<br>

- find로 얻은 결과도 BeutifulSoup 객체이다.
- 따라서 find를 한 결과에 또 find를 적용할 수 있다.

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(open('index.html'), "html.parser")

#find로 얻은 결과도 BeutifulSoup객체이다.
#따라서 find를 한 결과에 또 find를 적용할 수 있다.

soup.find("div", class_="elice").find("p")

#위 코드는 div태그 안에 있는 p태그를 추출한다.
```

<br>

- BeautifulSoup 객체에 get_text 메소드를 적용하면 태그가 갖고 있는 텍스트를 얻을 수 있다.

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(open('index.html'), "html.parser")

soup.find("div", class_="elice").find("p").get_text()

```

<br>

- html에서 특정 id의 값을 추출하고자 하는 경우에는 id매개변수의 값을 지정할 수 있다.

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(open('index.html'), "html.parser")

soup.find("div", id="elicd")
```

<br>
<br>

# 4. requests 라이브러리
requests 라이브러리: python에서 http요청을 보낼 수 있는 모듈이다.

<pre>
cf) http요청이란? 크게 2가지로 나눌 수 있다.
    ->GET요청: 정보를 조회하기 위한 요청 (ex) 네이버 홈페이지에 접속한다, 구글에 키워드를 검색한다.)
    ->POST 요청: 정보를 생성, 변경하기 위한 요청 (ex) 웹사이트에 로그인, 댓글달기, 메일을 삭제)
 크롤링 시 GET요청을 주로 사용
</pre>

## 1) GET요청
지정된 URL로 GET 요청을 보냈고, 서버에서는 요청을 받아 처리한 후 result변수에 응답을 보낸다.

```python
import requests
url = "https://www.google.com"
result = requests.get(url)

print(result.status_code) #status_code로 요청의 결과를 알 수 있다.
print(result.text) #만약 요청이 성공했다면 text로 해당 웹사이트의 HTML을 얻을 수 있다.
#전에 만든 requests와 BeutifulSoup를 조합하여 웹페이지의 HTML을 분석할 수 있다.
```

# 5. 실전 크롤링

**tip 1. 웹페이지에서 'F12'버튼을 눌러 개발자 도구를 켤 수 있다. -> 화면에 보이는 요소들의 html코드를 확인할 수 있다.** <br>
->웹페이지 구조 파악 가능, 찾고자하는 요소가 어떤 태그, 어떤 클래스인지 볼 수 있다. <br>
**tip 2. 검색을 원하는 요소에 오른쪽 마우스를 클릭하고 '검사'를 눌러 개발자 도구를 켤 수도 있다.**
**tip 3. find를 통헤 태그에 접근할 때 최대한 해당 태그에 바로 접근할 수 있게 태그를 설정해주는것이 좋다.*** 

## 1) 네이버 헤드뉴스 크롤링
**find_all()이후에 get_text()를 하면 안됨.** <br>
왜냐하면, find_all이 list를 반환해서 find-all.get_text()를 하면 html에서 get_text가 아닌 list에서 하는 것 이므로 <br>
반드시 list의 요소에 접근해서 get_text를 한다.

```python
import requests
from bs4 import BeautifulSoup

def crawling(soup):
    result = []

# 아래 코드를 참고하면 get_text()를 어떻게하는지 알 수 있다.
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    soup = soup.find("div", id="container").find("div", class_="column_left").find("div", id="newsstand").find("div", class_="issue_area").find("div").find_all("a")
    for i in soup: #list요소 하나하나에 있는 html코드를 꺼내면서, 거기에 있는 text를 받아온다.
        result.append(i.get_text())
    print(soup)
    return result

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    url = "http://www.naver.com"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser") #BeutifulSoup(가져올 코드, 코드의 형식(여기서는 html))
    print(crawling(soup))

if __name__ == "__main__" :
    main()
```

## 2) 연합뉴스 속보 크롤링

```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    # soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.
    list = []
    soup = soup.find("body").find("div",id="wrap").find("dl",class_="type04").find_all("a")
    for i in soup:
        list.append(i.get_text())
    list.remove('\n')
    return list


def main():
    url = "https://news.naver.com/main/list.nhn?mode=LPOD&mid=sec&sid1=001&sid2=140&oid=001&isYeonhapFlash=Y"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    print(crawling(soup))


if __name__ == "__main__":
    main()
```

## 3) bugs 음원 차트 크롤링

- str.replace("a","b"): 문자열 str에 등장하는 "a"란 문자를 모두 "b"로 대체하는 함수수
-> 개행문자 / 필요 없는 문자를 없애 필요한 데이터만 추출하는데 도움을 준다.
-> 자주 사용된다.

```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    # soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.
    list = []
    soup = soup.find("div",id="wrap").find("tbody").find_all("p", class_="title")
    for name in soup:
        list.append(name.get_text().replace("\n","")) #개행문자 제거

#cf) str.replace("a","b"): 문자열 str에 등장하는 "a"란 문자를 모두 "b"로 대체하는 함수수
    return list


def main():
    url = "https://music.bugs.co.kr/chart"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    # crawling 함수의 결과를 출력합니다.
    print(crawling(soup))


if __name__ == "__main__":
    main()
```

## 4) 네이버 영화 평가 크롤링

```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    list =[]
    soup = soup.find("ul",class_="rvw_list_area").find_all("strong")
    for review in soup:
        list.append(review.get_text())
    return list


def main():
    url = "https://movie.naver.com/movie/bi/mi/review.nhn?code=168058#"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    # crawling 함수의 결과를 출력합니다.
    for i in crawling(soup):
        print(i)


if __name__ == "__main__":
    main()
```

## 5) 커뮤니티 사이트에서 댓글 크로링
```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    list =[]
    soup = soup.find("div",class_="viewarea").find_all("dd",class_="usertxt")
    for i in soup:
        temp = i.find("span")
        list.append(temp.get_text().replace("\t","").replace("\n",""))

    return list

# soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.


def main():
    url = "https://pann.nate.com/talk/350939697"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    # crawling 함수의 결과를 출력합니다.
    for i in crawling(soup):
        print(i)


if __name__ == "__main__":
    main()
```

<br>
<br>

# 6. 여러페이지 크롤링하기

## 1) Query
한 뉴스 웹사이트는 각 페이지의 URL에서 'p=(숫자)' 부분이 20씩 증가하고 있는 규칙이 있다.<br>
이 사이트에서 여러 페이지를 크롤링하려면 어떻게 해야 할까?

### 1)) 쉬운 방법
쉬운 방법으로는 URL을 문자열 연산으로 처리하여 새로운 URL을 얻는 것이다. <br>
ex) <br>

```python
for i in range(0,5):
    url = "http://www.naver.com/Enter?p="+str((i*20+1))
```

### 2)) Query를 이용한 더 효과적인 방법
->URL의 query를 이용하면 이 작업을 더 효과적으로 할 수 있다. <br>

<br>

- Query의 정의: 웹서버에 GET 요청을 보낼 때, 조건에 맞는 정보를 표현하기 위한 변수
ex) 번호가 1번인 학생을 보여줘라(번호가 변수, 1이 값), <br>
&#160;전체 기사 중 페이지가 21인 기사들을 보여줘라(페이지가 변수, 21이 값)

















































































































