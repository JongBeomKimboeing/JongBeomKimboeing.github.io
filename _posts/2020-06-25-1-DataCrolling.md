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

<br>

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


<br>

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

<br>

## 3) bugs 음원 차트 크롤링

- str.replace("a","b"): 문자열 str에 등장하는 "a"란 문자를 모두 "b"로 대체하는 함수수<br>
-> 개행문자 / 필요 없는 문자를 없애 필요한 데이터만 추출하는데 도움을 준다.<br>
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

<br>

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

<br>

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


<br>

### 1)) 쉬운 방법
쉬운 방법으로는 URL을 문자열 연산으로 처리하여 새로운 URL을 얻는 것이다. <br>
ex) <br>

```python
for i in range(0,5):
    url = "http://www.naver.com/Enter?p="+str((i*20+1))
```


<br>

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


<br>

## 2) request 라이브러리 사용하여 쿼리 지정
request의 get 메소드로 GET 요청을 보낼 때 params 매개변수에 딕셔너리를 전달함으로서 쿼리를 지정할 수 있다.

```python
import requests
url = "https://www.google.com/search"
result = requests.get(url, params={'q':'elice'})
```


<br>


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


<br>

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

<br>

## 2) children, name (html태그의 속성)


<br>

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


<br>

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

<br>

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

<br>

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

<br>

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

<br>

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

<br>

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

<br>

## 6) 특정 영화 제목을 입력하면 영화제목 가져오기

- 문자열 사이에 변수를 넣기 위해서는 문자열 다온표 앞에 f를 붙인 다음 {}를 이용하여 변수이름을 적어주면 된다.<>
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

<br>
<br>
<br>

# 9. API
## 1) API의 정의

API(Application Programming Interface)는<br>
어떤 프로그램과 또 다른 프로그램을 연결해주는 매개체이다.<br>
사람이 컴퓨터를 다루기 위해 마우스와 키보드를 이용하는 것처럼,<br>
API는 프로그램 사이를 연결해주는 역할을 한다.<br>

ex) 지도 데이터를 이용하여 맛집 찾기 웹 서비스를 제작하려면 어떻게 할까?(지도 데이터와 웹서비스가 연결 해 줄 매개체가 없다)<br>
<br>
    보통의 일반인들에게는 지도 데이터를 갖고 있지 않고, 이를 수집하는 것도 매우 어렵다.<br>
    그렇다고 공개된 데이터를 그대로 사용하는 것도 어렵다.(호환성 문제, 만들고자하는 프로그램과 맞는 데이터가 없을 수 있다.)<br>
<br>
    만약, google이 갖고 있는 지도 데이터를 공개하였다고 가정하자.<br>
    그러나 구글지도 원본 데이터는 너무 방대하기도 하고, 호환성 등의 문제도 있어 쉽게 사용할 수 없다.<br>
    그래서 google은 지도 데이터를 응용하여 사용할 수 있도록 google map API라는 매개체를 사용자들에게 제공한다.<br>
  <br>  
  <br>
  <br>
  
    
daum 증권 사이트의 경우 여러 기업들의 주가 정보를 API를 거쳐 받아온 후 표시한다.<br>
daum 증권 사이트와 같이 API를 이용하여 정보를 가져오는 웹사이트가 꽤 많다.<br>
이런 경우 정보가 HTML에 처음부터 존재하지 않고, 정보를 API로부터 불러오고 나서 HTML에 존재하게 된다.<br>
그러므로, daum 증권 사이트에서는 BeautifulSoup를 이용하여 주가 데이터를 크롤링할 수 없다.<br>
왜냐하면 웹 사이트를 처음 로드할 때 HTML문서에는 주가 데이터가 존재하지 않기 때문이다.<br>
<br>
<br>
보통 API를 이용하여 데이터를 불러오는 경우는 데이터가 '동적'으로 변화하는 일이 많아<br>
실시간으로 값을 불러와야 하는 경우이다. (기업의 주가도 한 예시이다.)<br>
<br>
이럴 땐 daum 증권 사이트에서 주가 정보를 요청하는 API에 접근하여 어떤 정보를 전달해주고 있는지 접근하면 된다.<br>

<br>

<br>

## 2) API 데이터 가져오는 방법


크롬 개발자 도구(F12)의 Network 탭에서 웹사이트가 데이터를 요청하는 API를 볼 수 있다.<br>
(즉,네트워크를 통해서 외부 API에서 주가 데이터를 받아와서 웹페이지에 표시를 함)<br>
<br>
API의 URL에 GET 요청을 보내면 JSON 데이터를 얻을 수 있다.<br>
JSON은 key와 value를 저장하는, 딕셔너리 꼴의 데이터 형식이다.<br>
<br>
몇몇 웹사이트들은 크롤러 등을 통한 기계적인 접근을 막고 있다.<br>
이를 우회하기 위해 requests.get 메소드에 "headers" 매개변수를 지정해주어야 한다.<br>

<pre>
cf) header: http 상에서 클라이언트와 서버가 요청 또는 응답을 보낼 때 전송하는 부가적인 정보를 의미한다.
            실습에서 headers에 사용할 옵션을 제공하고 있다.

ex) custom_header = {
        'referer': ... (referer: 사용자가 해당 웹사이트에 접근할 때 이전 웹 페이지의 주소를 의미한다.)
        'user-agent': ... }(user-agent: 이용자의 여러가지 사양(브라우저, os)을 의미한다.)
    이렇게 custom_header를 만들어서 request가 get요청을 보낼 때 url과 함께 header 데이터도 같이 보낸다.

</pre>

<br>
<br>
<br>

## 3) API를 이용한 실전 크롤링

<br>

### 1)) daum 증권 페이지에서 주가 크롤링

- def get_data(json_obj): 와 def get_json(): 를 중심으로 보자.

```python
import requests
import json

custom_header = {
    'referer': 'http://http://finance.daum.net/quotes/A048410#home',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}


def get_data(json_obj):
    # 튜플들을 담고 있는 리스트를 반환해야 합니다.
    # 하나의 튜플은 한 기업의 (rank, name, tradeprice)를 담고 있어야 합니다.
    result = []
    for d in json_obj['data']:
        result.append((d['rank'], d['name'], d['tradePrice']))
        # API에 접속에 성공하였다면 json_obj 변수에 결과를 저장하세요.
    return result


def get_json():
    json_obj = None# 반환할 json 파일을 담는 변수입니다.
    url = "http://finance.daum.net/api/search/ranks?limit=10"  # 상위 10개 기업의 정보가 담긴 json 파일을 얻는 API url을 작성하세요.

    req = requests.get(url, headers=custom_header)

    if req.status_code == requests.codes.ok:
        print("접속 성공")
        #print(req.text) #request한 데이터를 출력
        json_obj = json.loads(req.text) #json 파일을 파이썬 코드에서 불러오기 위해 파이썬의 json모듈을 사용할 수 있다.
        #json.loads는 json 데이터를 로드하는 데 쓰인다.(핵심)
        #print(stock_data)
    else:
        print("접속 실패")

    return json_obj


def main():
    json_obj = get_json()
    data = get_data(json_obj)

    for d in data:
        print(d)


if __name__ == "__main__":
    main()

```

<br>

### 2)) 네이버 실시간 검색어 크롤링 (get_keyword_ranking함수를 이용하여 연관검색어 반환)

- def get_keyword_ranking(json_obj): 와 def get_data(json_obj, keyword_rank): 를 중심으로 보자

```python
import requests
import json

custom_header = {
    'referer': 'https://www.naver.com/',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}

def get_keyword_ranking(json_obj):
    key_rank=[]
    for data in json_obj['data']:
        key_rank.append(data['keyword_synonyms'])
    return key_rank

def get_data(json_obj, keyword_rank):
    # 튜플들을 담고 있는 리스트를 반환해야 합니다.
    # 하나의 튜플은 한 키워드의 (rank, keyword, keyword_synonyms)를 담고 있어야 합니다.
    result = []
    for data, key in zip(json_obj['data'], keyword_rank): #zip 이용하여 두 데이터 묶어주기
        result.append((data['rank'], data['keyword'],key))
    return result


def get_json():
    json_obj = None
    url = "https://apis.naver.com/mobile_main/srchrank/srchrank?frm=main&ag=20s&gr=2&ma=-2&si=1&en=-2&sp=-2"
    req = requests.get(url, headers=custom_header)

    if req.status_code == requests.codes.ok:
        print("접속 성공")
        json_obj = json.loads(req.text)
        # API에 접속에 성공하였다면 json_obj 변수에 결과를 저장하세요.

    else:
        print("Error code")

    return json_obj


def main():
    json_obj = get_json()
    keyword_rank = get_keyword_ranking(json_obj)
    result = get_data(json_obj, keyword_rank)

    for rank, keyword, synonyms in result:
        if synonyms:
            print(f"{rank}번째 검색어 : {keyword}, 연관검색어 : {synonyms}")
        else:
            print(f"{rank}번째 검색어 : {keyword}")


if __name__ == "__main__":
    main()
```

<br>

### 3)) 네이버 실시간 검색어 크롤링(get_keyword_ranking함수 없이)

- 위의 def get_keyword_ranking(json_obj): 없이도 크롤링 가능하다.

```python
import requests
import json

custom_header = {
    'referer': 'https://www.naver.com/',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}


def get_data(json_obj):
    # 튜플들을 담고 있는 리스트를 반환해야 합니다.
    # 하나의 튜플은 한 키워드의 (rank, keyword, keyword_synonyms)를 담고 있어야 합니다.
    result = []
    for data in json_obj['data']:
        result.append((data['rank'], data['keyword'],data['keyword_synonyms']))
    return result


def get_json():
    json_obj = None
    url = "https://apis.naver.com/mobile_main/srchrank/srchrank?frm=main&ag=20s&gr=2&ma=-2&si=1&en=-2&sp=-2"
    req = requests.get(url, headers=custom_header)

    if req.status_code == requests.codes.ok:
        print("접속 성공")
        json_obj = json.loads(req.text)
        # API에 접속에 성공하였다면 json_obj 변수에 결과를 저장하세요.

    else:
        print("Error code")

    return json_obj


def main():
    json_obj = get_json()
    result = get_data(json_obj)

    for rank, keyword, synonyms in result:
        if synonyms:
            print(f"{rank}번째 검색어 : {keyword}, 연관검색어 : {synonyms}")
        else:
            print(f"{rank}번째 검색어 : {keyword}")


if __name__ == "__main__":
    main()
```

<br>

### 4)) 음식점 리뷰 크롤링

```python
from bs4 import BeautifulSoup
import requests
import json

custom_header = {
    'referer': 'https://www.mangoplate.com',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}


def get_data(json_obj):
    # json_obj에 들어있는 댓글 텍스트를 담고 있는 리스트를 반환해야 합니다.
    result = []
    for data in json_obj:
        comment = data["comment"]  # comment 딕셔너리에 comment가 들어있다.
        text = comment["comment"]
        result.append(text)
    return result


def get_json(href, i):
    json_obj = None
    url = f"https://stage.mangoplate.com/api/v5{href}/reviews.json?language=kor&device_uuid=V3QHS15862342340433605ldDed&device_type=web&start_index={i}&request_count=50&sort_by=2"

    req = requests.get(url, headers=custom_header)

    if req.status_code == requests.codes.ok:
        print("접속 성공")
        json_obj = json.loads(req.text)

        # API에 접속에 성공하였다면 json_obj 변수에 결과를 저장하세요.

    else:
        print("Error code")

    return json_obj


def main():
    href = "/restaurants/iMRRP69qtkeO"  # 크롤링할 음식점의 고유 번호입니다.
    comments = []  # 음식점의 모든 댓글이 담길 리스트입니다.
    i = 0

    # 댓글을 모두 크롤링할 때 까지 계속 반복합니다.
    while True:
        json_obj = get_json(href, i)
        new_data = get_data(json_obj)

        if len(new_data) == 0:
            break

        comments += new_data

        i += 50

    print(comments)


if __name__ == "__main__":
    main()
```

<br>

### 5)) 음식점 href 크롤링

```python
from bs4 import BeautifulSoup
import requests
import json  # json import하기

custom_header = {
    'referer': 'https://www.mangoplate.com/',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}


def get_restaurants(name):
    # 검색어 name이 들어왔을 때 검색 결과로 나타나는 식당들을 리스트에 담아 반환하세요.
    restuarant_list = []
    url = f"https://www.mangoplate.com/search/{name}"
    req = requests.get(url, headers = custom_header)
    soup = BeautifulSoup(req.text,"html.parser")
    rdatas = soup.find_all("div",class_="list-restaurant-item")

    for rdata in rdatas:
        info = rdata.find("div", class_="info")
        link = info.find("")['href']
        names = info.find("h2", class_="title").get_text().replace("\n","").replace(" ","")
        restuarant_list.append([names,link])


    return restuarant_list


def main():
    name = input()
    restuarant_list = get_restaurants(name)
    print(restuarant_list)


if __name__ == "__main__":
    main()
```

<br>

### 6)) 검색 결과 음식점 리뷰 크롤링

- JSON으로 저장된 데이터에서 댓글을 추출하는 부분을 중심으로 (def get_reviews(code): )

```python
from bs4 import BeautifulSoup
import requests
import json  # json import하기

custom_header = {
    'referer': 'https://www.mangoplate.com/',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}


def get_reviews(code):
    comments = []
    url = f"https://stage.mangoplate.com/api/v5{code}/reviews.json?language=kor&device_uuid=V3QHS15862342340433605ldDed&device_type=web&start_index=0&request_count=5&sort_by=2"
    req = requests.get(url, headers=custom_header)
    json_obj = json.loads(req.text)

    if req.status_code == requests.codes.ok:
        print("접속 성공")
        for json_c in json_obj:
            comments.append(json_c["comment"]["comment"].replace("\n", ""))
    return comments
    # req에 데이터를 불러온 결과가 저장되어 있습니다.
    # JSON으로 저장된 데이터에서 댓글을 추출하여 comments에 저장하고 반환하세요.


def get_restaurants(name):
    name_links =[]
    url = f"https://www.mangoplate.com/search/{name}"
    req = requests.get(url, headers=custom_header)
    soup = BeautifulSoup(req.text, "html.parser")
    name_link = soup.find_all("div",class_="list-restaurant-item")
    for links in name_link:
        info = links.find("div",class_="info")
        names = info.find("h2",class_="title").get_text().replace("\n","").replace(" ","")
        link = info.find("a")["href"]
        name_links.append((names, link))
    return name_links
    # soup에는 특정 키워드로 검색한 결과의 HTML이 담겨 있습니다.
    # 특정 키워드와 관련된 음식점의 이름과 href를 튜플로 저장하고,
    # 이름과 href를 담은 튜플들이 담긴 리스트를 반환하세요.


def main():
    name = input("검색어를 입력하세요 : ")

    restuarant_list = get_restaurants(name)

    for r in restuarant_list:
        print(r[0])
        print(get_reviews(r[1]))
        print("=" * 30)
        print("\n" * 2)


if __name__ == "__main__":
    main()
```

<br>
<br>
<br>

# 10. 워드 클라우드

<pre>
1. 워드클라우드의 정의
   -> 워드클라우드란, 데이터에서 단어 빈도를 분석하여 시각화하는 기법이다.
   
2. 워드클라우드 준비
   -> 워드클라우드를 그리기 위해서 텍스트 데어터가 필요하다.
   
3. 영어 문장 나누기
   ->워드클라우드의 각 단어는 빈도에 따라 크기가 결정된다.
   크기가 큰 단어일수록 빈도가 놓다.
   영어 문장의 경우, 공백을 기준으로 나누어 각각의 단어를 얻을 수 있다.
   (한국어의 경우 조사가 있어서 공백으로 나눈다고 해도 올바르게 단어를 구분할 수 없다.)

</pre>

<br>

## 1) 영어 문장 나누기

- 주석을 따라가 보자.

```python
from collections import Counter # Counter: 주어진 리스트에서 특정 값이 몇 번 등장하는지 세는 역할을 한다.
from string import punctuation # punctuation(특수문자들이 담긴 문자열): 문자열 데이터에서 특수문자를 제거하기위해 사용한다.
from text import data

def count_word_freq(data):
    _data = data.lower() # 문자열들을 모두 소문자로 바꾸어 전처리한다.
    for p in punctuation: # 문자열에 들어있는 특수문자를 모두 제거한다.
#                           paunctuation은 특수문자들이 담겨있는 문자열 변수이다.
        _data = _data.replace("p","")
    _data = _data.split() # 공백을 기준으로 데이터 나누기
    counter = Counter(_data) # 단어 count하기
    return counter

if __name__ == "__main__":
    print(count_word_freq(data))
```

<br>

## 2) 워드클라우드 출력하기

- def create_word_cloud(data): 를 중심으로 보기

```python
from wordcloud import WordCloud
from string import punctuation
from collections import Counter
from text import data
import matplotlib.pyplot as plt

def count_word_freq(data):
    _data = data.lower() # 문자열들을 모두 소문자로 바꾸어 전처리한다.
    for p in punctuation: # 문자열에 들어있는 특수문자를 모두 제거한다.
#                           paunctuation은 특수문자들이 담겨있는 문자열 변수이다.
        _data = _data.replace("p","")
    _data = _data.split() # 공백을 기준으로 데이터 나누기
    counter = Counter(_data) # 단어 count하기
    return counter

def create_word_cloud(data):
    counter = count_word_freq(data)
    cloud = WordCloud(background_color='white') #배경이 흰색인 wordcloud 객체 생성
    cloud.fit_words(counter) #단어들의 횟수를 기반으로 워드클라우드 생성
    plt.imshow(cloud)
    plt.show()
    # 코드를 작성하세요.

    return None


if __name__ == "__main__":
    create_word_cloud(data)
```

<br>
<br>


### 이전 실습의 문제점
이전 실습에서 그렸던 워드클라우드의 문제점은 단어에 어미와 조사가 붙어 분석이 왜곡되는 것이다.<br>
ex) '대통령이'와 '대통령은'은 둘 다 대통령이라는 공통된 키워드로 집계되어야 한다.<br>
이를 추출하기 위해 한국어 단어에 붙는 어미와 조사를 제거하고, 단어의 어근만 집계되도록 하는 형태소 추출 과정이 필요하다.<br>

<br>

## 3) 형태소 추출 없이 네이버 기사로 워드클라우드 만들기

- 아래 코드는 형태소 추출을 하지 않았다.
- 한글을 출력하기 위해서는 NanumBarunGothic.ttf 가 필요하다. (이를 중심으로 보자)

```python
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt

def count_word_freq(text):
    text = text.lower()
    for p in punctuation:
        text = text.replace("p","")
    text = text.split() # 띄어쓰기 기준으로 단어 나누기
    counter = Counter(text)
    return counter

def create_word_cloud(text):
    counter = count_word_freq(text)
    cloud = WordCloud(font_path='C:/Users/harry/NanumBarunGothic.ttf',background_color='white')
    cloud.fit_words(counter)
    cloud.to_file('cloud.png')

def crawling(soup):
    # soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.
    text_list = ""
    children = soup.find("div",class_="_article_body_contents").children
    text = soup.find("div",class_="_article_body_contents")
    for child in children:
        if child.name == None:
            text_list += child
    start = text_list.find("// TV플레이어")
    text_list = text_list[start + len("// TV플레이어")+1:] #"// TV플레이어"가 적힌 곳 부터 분문 끝까지 추출
    end = text_list.find("// 본문 내용 ")
    text_list = text_list[:end]
    text_list.replace("\n", "").replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}', '').replace("\t", "")
    return text_list

def main():
    url = "https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=100&oid=005&aid=0001328950"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    text = crawling(soup)
    create_word_cloud(text)


if __name__ == "__main__":
    main()
```


<br>

## 4) mecab을 이용하여 형태소 추출을 해낸다.

- text = hannanum.nouns(text) 이 코드 한 줄을 이용하면 형태소 추출을 할 수 있다.

```python
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt
from konlpy.tag import Hannanum
hannanum = Hannanum()

def count_word_freq(text):
    text = text.lower()
    for p in punctuation:
        text = text.replace("p","")
    text = hannanum.nouns(text)# 띄어쓰기 기준으로 단어 나누기
    counter = Counter(text)
    return counter

def create_word_cloud(text):
    counter = count_word_freq(text)
    cloud = WordCloud(font_path='C:/Users/harry/NanumBarunGothic.ttf',background_color='white')
    cloud.fit_words(counter)
    cloud.to_file('cloud.png')

def crawling(soup):
    # soup 객체에서 추출해야 하는 정보를 찾고 반환하세요.
    text_list = ""
    children = soup.find("div",class_="_article_body_contents").children
    text = soup.find("div",class_="_article_body_contents")
    for child in children:
        if child.name == None:
            text_list += child
    start = text_list.find("// TV플레이어")
    text_list = text_list[start + len("// TV플레이어")+1:] #"// TV플레이어"가 적힌 곳 부터 분문 끝까지 추출
    end = text_list.find("// 본문 내용 ")
    text_list = text_list[:end]
    text_list.replace("\n", "").replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}', '').replace("\t", "")
    return text_list

def main():
    url = "https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=100&oid=005&aid=0001328950"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    text = crawling(soup)
    create_word_cloud(text)


if __name__ == "__main__":
    main()
```

<br>
<br>
<br>


# 11. 실전 데이터 크롤링 / 워드클라우드
- 앞에서 배운 모든 내용을 활용해 실습을 해보자.

<br>
<br>


## 1) 여러 개의 기사 내용 크롤링하기

하나의 기사만으로는 단어의 빈도수를 파악하기 어려울 수 있다. 기사의 분량, 기자의 성향 등 여러 요인이 반영되기 때문이다.<br>
그러므로, 공통된 주제에 대한 여러 기사의 텍스트 데이터를 같이 분석하면 효과적인 워드클라우드를 출력할 수 있다.  


```python
import requests
from bs4 import BeautifulSoup



def crawling(soup):
    # 기사에서 내용을 추출하고 반환하세요.

    div = soup.find('div', class_="_article_body_contents")
    result = div.get_text().replace('\n', '').replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}',
                                                      '').replace('\t', '')

    return result


def get_href(soup):
    result = []
    hiper = soup.find_all("div",class_="cluster_text")
    for links in hiper:
        result.append(links.find("a")['href'])
    return result


def get_request(section):
    # 입력된 분야에 맞는 페이지의 URL을 반환합니다.
    url = "https://news.naver.com/main/main.nhn"
    section_dict = {"정치": 100,
                    "경제": 101,
                    "사회": 102,
                    "생활": 103,
                    "세계": 104,
                    "과학": 105}
    return requests.get(url, params={"sid1": section_dict[section]})


def main():
    list_href = []
    result = []

    # 섹션을 입력하세요.
    section = input('"정치", "경제", "사회", "생활", "세계", "과학" 중 하나를 입력하세요.\n  > ')

    req = get_request(section)
    soup = BeautifulSoup(req.text, "html.parser")

    list_href = get_href(soup)

    for href in list_href:
        href_req = requests.get(href)
        href_soup = BeautifulSoup(href_req.text, "html.parser")
        result.append(crawling(href_soup))
    print(result)


if __name__ == "__main__":
    main()
```

<br>
<br>

## 2) 여러 개의 기사 내용으로 워드클라우드 출력하기

- 형태소 분석을 하고 단어 빈도수를 count 했다.
- 이를 이용하여 워드클라우드도 출렸했다.

```python
import requests
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
import mecab
mecab = mecab.MeCab()

def count_word_freq(data):
    _data = data.lower()

    for p in punctuation:
        _data = _data.replace(p, "")

    # 명사 추출
    _data = mecab.nouns(_data)

    counter = Counter(_data)

    return counter


def crawling(soup):
    result = ""
    for children in soup.find("div", class_="_article_body_contents").children:
        if children.name == None:
            result += children

    start = result.find("// TV플레이어")
    result = result[start + len("// TV플레이어") + 1:]

    end = result.find("// 본문 내용")
    result = result[:end]

    return result.replace("\n", "").replace("\t", "")


def get_href(soup):
    result = []

    cluster_body = soup.find("div", class_="cluster_body")

    for cluster_text in cluster_body.find_all("div", class_="cluster_text"):
        result.append(cluster_text.find("a")["href"])

    return result


def get_request(section):
    url = "https://news.naver.com/main/main.nhn"
    section_dict = {"정치": 100,
                    "경제": 101,
                    "사회": 102,
                    "생활": 103,
                    "세계": 104,
                    "과학": 105}
    return requests.get(url, params={"sid1": section_dict[section]})


def main():
    list_href = []
    result = []

    # 섹션을 입력하세요.
    section = input('"정치", "경제", "사회", "생활", "세계", "과학" 중 하나를 입력하세요.\n  > ')

    req = get_request(section)
    soup = BeautifulSoup(req.text, "html.parser")

    list_href = get_href(soup)

    for href in list_href:
        href_req = requests.get(href)
        href_soup = BeautifulSoup(href_req.text, "html.parser")
        result.append(crawling(href_soup))

    text = " ".join(result)
    create_word_cloud(text)


if __name__ == "__main__":
    main()
```

<br>
<br>

## 3) 더 많은 기사 내용 크롤링하기

```python
import requests
from bs4 import BeautifulSoup


def crawling(soup):
    # 기사에서 내용을 추출하고 반환하세요.
    div = soup.find('div', class_="_article_body_contents")

    result = div.get_text().replace('\n', '').replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}',
                                                      '').replace('\t', '')

    return result


def get_href(soup,section):
    # 분야별 기사 페이지에서 최상단의 "주제"에 속한 기사들의 href 링크를 리스트에 담아 반환하세요.
    result = []
    if section == "정치":
        url = soup.find("div",class_="cluster_foot_inner").find("a",class_="cluster_foot_more")['href']
    else:
        url = soup.find("div", class_="cluster_head_inner").find("a")['href']
    req = requests.get("https://news.naver.com"+url)
    soup = BeautifulSoup(req.text,"html.parser")
    links = soup.find("div",class_="content").find_all("li")
    for link in links:
        result.append(link.find("dt").find("a")['href'])

    return result
def get_request(section):
    # 입력된 분야에 맞는 페이지의 URL을 반환합니다.
    url = "https://news.naver.com/main/main.nhn"
    section_dict = {"정치": 100,
                    "경제": 101,
                    "사회": 102,
                    "생활": 103,
                    "세계": 104,
                    "과학": 105}
    return requests.get(url, params={"sid1": section_dict[section]})


def main():
    list_href = []
    result = []

    # 섹션을 입력하세요.
    section = input('"정치", "경제", "사회", "생활", "세계", "과학" 중 하나를 입력하세요.\n  > ')

    req = get_request(section)
    soup = BeautifulSoup(req.text, "html.parser")

    list_href = get_href(soup,section)


    for href in list_href:
        href_req = requests.get(href)
        href_soup = BeautifulSoup(href_req.text, "html.parser")
        result.append(crawling(href_soup))
    print(result)


if __name__ == "__main__":
    main()
```

<br>
<br>

## 4) 더 많은 기사로 워드클라우드 출력하기

- 위 코드에서 형태소 분석, 단어 count, 워드클라우드를 추가했다.

```python
import requests
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
import mecab
from wordcloud import WordCloud

mecab = mecab.MeCab()


def create_word_cloud(data):
    counter = count_word_freq(data)

    cloud = WordCloud(font_path='NanumBarunGothic.ttf', background_color='white')
    cloud.fit_words(counter)
    cloud.to_file('cloud.png')


def count_word_freq(data):
    _data = data.lower()

    for p in punctuation:
        _data = _data.replace(p, "")

    # 명사 추출
    _data = mecab.nouns(_data)

    counter = Counter(_data)

    return counter


def crawling(soup):
    # 기사에서 내용을 추출하고 반환하세요.
    result = ""
    for children in soup.find("div", class_="_article_body_contents").children:
        if children.name == None:
            result += children

    start = result.find("// TV플레이어")
    result = result[start + len("// TV플레이어") + 1:]

    end = result.find("// 본문 내용")
    result = result[:end]

    return result.replace("\n", "").replace("\t", "")


def get_href(soup,section):
    # 분야별 기사 페이지에서 최상단의 "주제"에 속한 기사들의 href 링크를 리스트에 담아 반환하세요.
    result = []
    if section == "정치":
        url = soup.find("div",class_="cluster_foot_inner").find("a",class_="cluster_foot_more")['href']
    else:
        url = soup.find("div", class_="cluster_head_inner").find("a")['href']
    req = requests.get("https://news.naver.com"+url)
    soup = BeautifulSoup(req.text,"html.parser")
    links = soup.find("div",class_="content").find_all("li")
    for link in links:
        result.append(link.find("dt").find("a")['href'])

    return result
def get_request(section):
    # 입력된 분야에 맞는 페이지의 URL을 반환합니다.
    url = "https://news.naver.com/main/main.nhn"
    section_dict = {"정치": 100,
                    "경제": 101,
                    "사회": 102,
                    "생활": 103,
                    "세계": 104,
                    "과학": 105}
    return requests.get(url, params={"sid1": section_dict[section]})


def main():
    list_href = []
    result = []

    # 섹션을 입력하세요.
    section = input('"정치", "경제", "사회", "생활", "세계", "과학" 중 하나를 입력하세요.\n  > ')

    req = get_request(section)
    soup = BeautifulSoup(req.text, "html.parser")

    list_href = get_href(soup,section)


    for href in list_href:
        href_req = requests.get(href)
        href_soup = BeautifulSoup(href_req.text, "html.parser")
        result.append(crawling(href_soup))
    text = " ".join(result)
    create_word_cloud(text)


if __name__ == "__main__":
    main()
```











































