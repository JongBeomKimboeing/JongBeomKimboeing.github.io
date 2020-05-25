# 카드2 (2164번)

STL을 이용하여 아주 간단하고 쉽게 풀었다.

```c
#include<stdio.h>
#include<queue>
using namespace std;
queue <int> q;

int main(void) {
	int num = 0;
	scanf("%d", &num);
	for (int i = 1; i <= num; i++) {
		q.push(i);
	}
	while (q.size() > 1) {
		q.pop();
		int second = q.front();
		q.pop();
		q.push(second);
	}
	printf("%d",q.front());
}
```

#요세푸스문제 (1158번)

이번 문제도 어렵지 않게 풀었다.
역시 STL은 신세계이다.

```c
#include<stdio.h>
#include <queue>
using namespace std;
queue <int> q;

int main(void) {
	int N = 0;
	int K = 0;
	scanf("%d", &N);
	scanf("%d", &K);
	for (int person = 1; person <= N; person++) {
		q.push(person);
	}
	printf("<");
	while (!q.empty()) {
		for (int k = 1; k < K; k++) {
			q.push(q.front());
			q.pop();
			}
		printf("%d", q.front());
		if (q.size() == 1) {
			printf("");
		}
		else {
			printf(", ");
		}
		q.pop();
		}
	printf(">");
}
```
