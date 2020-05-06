# 5월 4일(3문제)
## 1850번(최대공약수)
```c
#include<stdio.h>
#include<math.h>

int Getnum(int a) {
	int num = 0;
	while (a) {
		a--;
		num = num * 10 + 1;
	}
	return num;
}


long long int minmul(long long int n,long long int m) {
	if (m % n == 0) {
		return n;
	}
		m = n * (m / n) + (m % n);
	minmul(m % n, n);
}
int main(void) {
	int input = 0;
		long long int n = 0, m = 0;
		//long long int k = 0, c = 0;
		scanf("%lld %lld", &n, &m);
		for(int i=0; i< minmul(n, m); i++)
			printf("1");
	
}
```
## 1157번(단어 공부)
계속 실패 중(시간초과)
```c
#include<stdio.h>
int main(void) {
	int i = 0,cnt = 0, cntb = 0, mem=0;
	char word[1000000] = { '\0' };
	scanf("%s", word);
	while (word[i] != '\0') {
		cnt = 0;
		if (word[i] <= 90) {
			word[i]=word[i] + 32;
		}
		for (int j = i; word[j] != '\0'; j++) {
			if (word[i] == word[j]) {
				cnt++;
			}
		}
		if (cntb <= cnt) {
			cntb = cnt;
			mem = i;
		}
		i++;
	}
	if (cntb == cnt) {
		printf("?");
	}
	printf("%c", word[mem]-32);
}
```
## 10872번(팩토리얼)
```c
#include<stdio.h>
int Fact(int num) {
	if (num > 0) {
		return num * Fact(num - 1);;
	}
	else {
		return 1;
	}

}
int main(void) {
	int input = 0;
	scanf("%d", &input);
	printf("%d", Fact(input));
}
```
