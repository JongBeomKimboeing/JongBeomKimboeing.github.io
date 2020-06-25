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
<br>

# 2. 크롤링을 위해 필요한 것
웹페이지의 HTML을 얻기 위해 request 라이브러리를 이용하고<br>
가져온 HTML을 분석하기 위해 BeutifulSoup 라이브러리를 사용한다.

<br>
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

<br>
<br>
<br>

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

- Query의 정의: 웹서버에 GET 요청을 보낼 때, 조건에 맞는 정보를 표현하기 위한 변수<br>
ex) 번호가 1번인 학생을 보여줘라(번호가 변수, 1이 값), <br>
ex) 전체 기사 중 페이지가 21인 기사들을 보여줘라(페이지가 변수, 21이 값) <br>

<br>

ex) https://www.google.com/search?q=elice <br>
위 예시는 google에 'elice'을 검색한 결과이다. q라는 변수에 elice라는 값이 담겨,<br>
전체 데이터 중 elice라는 키워드로 검색한 결과만을 보여준다. <br>

<br>

ex) https://movie.naver.com/movie/bi/mi/basic.nhn?code=168058<br>
네이버 영화 서비스에서 특정 영화를 클릭하면, code라는 변수에 영화 코드가 담겨 해당 영화에 대한 정보를 보여준다.(여기서는 168058) <br>

## 2) request 라이브러리 사용하여 쿼리 지정
request의 get 메소드로 GET 요청을 보낼 때 params 매개변수에 딕셔너리를 전달함으로서 쿼리를 지정할 수 있다.

```python
import requests
url = "https://www.google.com/search"
result = requests.get(url, params={'q':'elice'})
```
### cf) requests.get() 연속으로 하기
전체 영화 데이터에서 영화 코드에 대한 정보를 크롤링을 이용해서 찾고<br>
찾은 영화 코드를 다시 request를 이용하여 특정 영화에 대한 정보를 얻는 요청을 할 수 있다.<br>
이런 경우 requests.get 연산을 두 번 하게 된다.<br>

<br>

 ex)<br>
 만약 https://movie.naver.com/movie/bi/mi/basic.nhn?code=168058 에서  <br>
 특정 영화에 대한 정보를 얻는 요청을 한다고 할 때<br>
 아래와 같이 code에 대한 requests를 한 뒤 한번 더 특정 영화에 대해 requests를 한다.
```python
code = ...
result = requests.get(url, params= {'movie':code})
```

<br>
<br>
<br>

# 7. 태그와 속성
HTML에는 여러 종류의 태그와, 태그에 특정 기능이나 유형을 적용하는 속성이 있다.<br>

<br>

ex) <br>
div는 태그, class와 id를 속성이라고 한다.
```html
<div class="elice" id="title">제목</div> <br>
```

- 어떤 태그의 속성이 무엇이 있는지 확인할 때는 attrs 멤버변수를 출력한다.

```python
div = soup.find("div") #find를 이용해 tag를 찾는다
print(div.attrs) #찾은 tag에 속성이 무엇이 있는지 확인 할 때는 attrs를 이용한다.
```

- attrs 딕셔너리의 키로 인덱싱하여, 태그의 속성에 접근할 수 있다.

```python

print(div['class']) #div 태그 안에 있는 class가 어떤 class인지 출력한다.

```

## 1) href 속성
- a 태그는 하이퍼링크를 걸어주는 태그로써 이동할 URL을 href속성에 담고있다.
```html

<a href="https...">기사 제목</a>

```
그러므로 어떠한 요소들 중 a라는 태그를 찾고 해당 a 태그의 href라는 속성을 얻음으로써 새로운 페이지의 url을 얻을 수 있다.

<br>

- 아래와 같이 href 속성을 이용하여 웹페이지에 존재하는 하이퍼링크의 URL을 얻을 수 있다.
```python
a = soup.find("a")
href_url = a["href"]
```

## 2) children, name (html태그의 속성)

### 1)) children

웹사이트의 구조가 복잡한 경우 다양한 옵션을 적용해서 html태그를 검색할 수도 있다.<br>
이때 children이라는 속성은 어떤 태그가 포함하고 있는 태그이다.<br>

<br>

ex)<br>
div안에 여러 요소들이 있다면 각각의 요소를 children이라고 한다.

```html
<div>
    <span>span1</span>
    <span>span1</span>
    <p>p tag</p>
    <img ... />
</div>
```
위의 div태그는 span,p,img 태그들을 갖고 있다.<br>
beautifulsoup의 children 속성으로 어떤 태그가 포함하고 있는 태그들도 조회할 수 있다.<br>

```python

soup.find("div").children
```
위의 코드는 어떤 div 태그를 찾고, 그 div 태그에 포함된 태그들의 리스트를 얻는 코드이다.<br>
위 코드를 통해 span,p,img 태그를 갖는 리스트를 얻을 수 있다.<br>

### 2)) name

name은 어떤 태그의 이름을 의미하는 속성이다.<br>

ex)<br>
div, span -> name이라는 속성은 div가 갖고 있는 여러가지 children이 각각 어떤 태그인 지 알 때 유용<br>

```html
<div>
    <span>span1</span>
    <span>span1</span>
    <p>p tag</p>
    <img ... />
</div>
```
어떤 태그의 이름을 알고 싶다면 name속성을 이용할 수 있다.<br>
태그가 존재하지 않는 경우 None 값을 얻는다.

```python
children = soup.find("div").children
for child in children:
    print(child.name)
```
결과: span,span,p,img가 각각 출력된다.

<br>
<br>
<br>


# 8. 실전 크롤링

## 1) 여러 페이지의 기사 제목 수집하기

- Query 사용을 중심으로 

```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    # soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.
    list =[]
    soup = soup.find("div", class_="sub_content").find_all("span", class_="tit")
    for data in soup:
        list.append(data.get_text())

    return list


def main():
    answer = []
    url = "https://sports.donga.com/ent"

    for i in range(0, 5):
        req =  requests.get(url, params={'p':(20*i+1)}) # requests.get 메소드를 호출해보세요.
        soup = BeautifulSoup(req.text, "html.parser")
        answer += crawling(soup)
    # crawling 함수의 결과를 출력합니다.
    print(answer)
if __name__ == "__main__":
    main()

```

## 2) 각 기사의 href 수집하기

- href 속성 가져오는 것을 중심으로 보기

```python
import requests
from bs4 import BeautifulSoup

def get_href(soup) :
    href_list = []
    # soup에 저장되어 있는 각 기사에 접근할 수 있는 href들을 담고 있는 리스트를 반환해주세요.
    a = soup.find("ul", class_="list_news").find_all('span',class_="tit")
    for data in a:
        href_list.append(data.find("a")["href"])
    return href_list

def main():
    list_href = []

    url = "https://sports.donga.com/ent?p=1&c=02"
    result = requests.get(url)
    soup = BeautifulSoup(result.text, "html.parser")
    list_href = get_href(soup)
    print(list_href)


if __name__ == "__main__":
    main()

```

## 3) 네이버 최신뉴스 href 수집하기

- find 를 이용해 div의 class 속성을 찾아줄 때는 최대한 안 쪽 클래스로 설정해준다.

```python
import requests
from bs4 import BeautifulSoup

def get_href(soup):
    list =[]
    # 각 기사에 접근할 수 있는 href를 리스트로 반환하세요.
    data = soup.find_all("div", class_="mlt01") #최대한 안 쪽 클래스로 설정해준다.
    for link in data:
        list.append("https:"+link.find("a")["href"])
    return list

def main():
    list_href = []

    # href 수집할 사이트 주소 입력
    url = "https://news.nate.com/recent?mid=n0100"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    list_href = get_href(soup)

    print(list_href)

if __name__ == "__main__":
    main()
```

## 4) 다양한 섹션의 속보기사 href 추출하기

- dictionary 로 Query를 적용한 부분을 중심으로

```python
import requests
from bs4 import BeautifulSoup


def get_href(soup):
    list_link =[]
    # 각 분야별 속보 기사에 접근할 수 있는 href를 리스트로 반환하세요.
    resultsoup = soup.find("ul", class_="type06_headline").find_all("a")
    for data in resultsoup:
        list_link.append(data['href'])
    return list_link

#----------------------------------------------------------------------------------
def get_request(section):
    # 입력된 분야에 맞는 request 객체를 반환하세요.
    # 아래 url에 쿼리를 적용한 것을 반환합니다.
    sec_dict = {"정치":100, "경제":101, "사회":102, "생활":103, "세계":104, "과학":105}
    url = "https://news.naver.com/main/list.nhn"
    result = requests.get(url,params={'sid1':sec_dict[section]})
    return result
#----------------------------------------------------------------------------------

def main():
    list_href = []

    # 섹션을 입력하세요.
    section = input('"정치", "경제", "사회", "생활", "세계", "과학" 중 하나를 입력하세요.\n  > ')

    req = get_request(section)
    soup = BeautifulSoup(req.text, "html.parser")
    list_href = get_href(soup)
    print(list_href)


if __name__ == "__main__":
    main()
```


## 5) 다양한 섹션의 속보 기사 내용 추출하기

- Naver 신문에서의 기사는 "div",id="articleBodyContents" 부분에 있는데, div의 children 속성 없는 곳에 기사가 적혀있다.
- def crawling(soup): 을 중심으로 보자.


```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    # 기사에서 내용을 추출하고 반환하세요.
    children = soup.find("div",id="articleBodyContents").children
    article = soup.find("div",id="articleBodyContents")
    for child in children:
        if child.name == None:
            list_ch = article.get_text().replace("\n","").replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}',"").replace("\t","")
    return list_ch


def get_href(soup):
    # 각 분야별 속보 기사에 접근할 수 있는 href를 리스트로 반환하세요.
    links=[]
    link_result = soup.find("ul","type06_headline").find_all("a")
    for link in link_result:
        links.append(link['href'])
    return links


def get_request(section):
    # 입력된 분야에 맞는 request 객체를 반환하세요.
    # 아래 url에 쿼리를 적용한 것을 반환합니다.
    sec_dict = {"정치":100, "경제":101, "사회":102, "생활":103, "세계":104, "과학":105}
    url = "https://news.naver.com/main/list.nhn"
    result = requests.get(url,params={'sid1':sec_dict[section]})
    return result


def main():
    list_href = []
    result = []

    # 섹션을 입력하세요.
    section = input('"정치", "경제", "사회", "생활", "세계", "과학" 중 하나를 입력하세요.\n  > ')

    req = get_request(section)
    soup = BeautifulSoup(req.text, "html.parser")

    list_href = get_href(soup)
    print(list_href)

    for href in list_href:
        href_req = requests.get(href)
        href_soup = BeautifulSoup(href_req.text, "html.parser")
        result.append(crawling(href_soup))
    print(result)


if __name__ == "__main__":
    main()
```

## 6) 특정 영화 제목을 입력하면 영화제목 가져오기

- 문자열 사이에 변수를 넣기 위해서는 문자열 다온표 앞에 f를 붙인 다음 {}를 이용하여 변수이름을 적어주면 된다.
(def get_url(movie): 을 중심으로 보자.)

```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    # soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.
    # 1장 실습의 영화 리뷰 추출 방식과 동일합니다.
    reply_list =[]
    reply = soup.find("ul",class_="rvw_list_area").find_all("strong")
    for replys in reply:
        reply_list.append(replys.get_text().replace("\t","").replace("\n","").replace("\r",""))
    return reply_list


def get_href(soup):
    # 검색 결과, 가장 위에 있는 영화로 접근할 수 있는 href를 반환하세요.
    result = soup.find("ul",class_="search_list_1").find("li").find("a")
    return "https://movie.naver.com"+result['href'].replace("basic","review")


def get_url(movie):
    # 입력된 영화를 검색한 결과의 url을 반환하세요.
    mreq = f"https://movie.naver.com/movie/search/result.nhn?query={movie}&section=all&ie=utf8"
    #문자열 사이에 변수를 넣기 위해서는 문자열 다온표 앞에 f를 붙인 다음 {}를 이용하여 변수이름을 적어주면 된다.
    return mreq


def main():
    list_href = []

    # 섹션을 입력하세요.
    movie = input('영화 제목을 입력하세요. \n  > ')

    url = get_url(movie)

    req = requests.get(url)

    soup = BeautifulSoup(req.text, "html.parser")
    movie_url = get_href(soup)
    print(movie_url)
    href_req = requests.get(movie_url)
    href_soup = BeautifulSoup(href_req.text, "html.parser")
    print(crawling(href_soup))



if __name__ == "__main__":
    main()
```



















































































