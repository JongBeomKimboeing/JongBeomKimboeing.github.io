---
layout: post
title: 5월 6일 문제풀이
description: "5월 6일 문제풀이"
modified: 2020-05-06
tags: [알고리즘,재귀함수,배열]
categories: [백준문제풀이]
---
# 5월 6일(4문제 1시도)
## 11778번(피보나치수와 최대공약수)
문제 미해결(메모리 초과)
```c
#include<stdio.h>
Fivo(long long int n) {
	if (n == 0) {
		return 0;
	}
	if (n == 1) {
		return 1;
	}
	return Fivo(n - 1) + Fivo(n - 2);
}

Gcd(long long int a, long long int b) {
	if (b % a == 0) {
		return a;
	}
	b = a * (b / a) + (b % a);
	Gcd(b % a, a);
}
int main(void) {
	long long int ain = 0, bin = 0;
	scanf("%lld %lld", &ain, &bin);
	ain = Fivo(ain);
	bin = Fivo(bin);
	int res = Gcd(ain,bin);
	//int Gld = (ain / res) * (bin / res) * res;
	printf("%d", res % 1000000007);	
}
```
## 113698번(Hawk eyes)
```c
#include<stdio.h>
void SwitchCup(int *cup, char sw) {
	int tmp = 0;
	if (sw == 'A') {
		tmp = cup[0];
		cup[0] = cup[1];
		cup[1] = tmp;
	}
	if (sw == 'B') {
		tmp = cup[0];
		cup[0] = cup[2];
		cup[2] = tmp;
	}
	if (sw == 'C') {
		tmp = cup[0];
		cup[0] = cup[3];
		cup[3] = tmp;
	}
	if (sw == 'D') {
		tmp = cup[1];
		cup[1] = cup[2];
		cup[2] = tmp;
	}
	if (sw == 'E') {
		tmp = cup[1];
		cup[1] = cup[3];
		cup[3] = tmp;
	}
	if (sw == 'F') {
		tmp = cup[2];
		cup[2] = cup[3];
		cup[3] = tmp;
	}
}
int main(void) {
	int cup[4] = { 1, 2, 3, 4 };
	char input[201] = { '\0' };
	scanf("%s",input);
	for (int i = 0; input[i] != '\0'; i++) {
		SwitchCup(cup, input[i]);
	}
	for (int j = 0; j < 4; j++) {
		if (cup[j] == 1) {
			printf("%d\n", j + 1);
		}
	}
	for (int j = 0; j < 4; j++) {
		if (cup[j] == 4) {
			printf("%d", j + 1);
		}
	}
		
}
```
## 1934번(최소공배수)
```c
#include<stdio.h>
Gcd(int a, int b) {
	if (b % a == 0) {
		return a;
	}
	b = a * (b / a) + (b % a);
	Gcd(b % a, a);
}
int main(void) {
	int ain=0, bin=0,num=0;
	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		scanf("%d %d", &ain, &bin);
		int res = Gcd(ain, bin);
		int Gld = (ain / res) * (bin / res) * res;
		printf("%d\n", Gld);
	}
}
```
## 2609번(최대공약수와 최소공배수)
```c
#include<stdio.h>
Gcd(int a, int b) {
	if (b % a == 0) {
		return a;
	}
	b = a * (b / a) + (b % a);
	Gcd(b % a, a);
}
int main(void) {
	int ain=0, bin=0,num=0;
		scanf("%d %d", &ain, &bin);
		int res = Gcd(ain, bin);
		int Gld = (ain / res) * (bin / res) * res;
		printf("%d\n", res);
		printf("%d", Gld);
}
```
## 1316번(그룹 단어 체커)
```c
#include<stdio.h>

int main(void) {
	int num = 0;
	int cnt = 0;
	int mcnt = 0;
	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		mcnt = cnt;
		char word[101] = { '\0' };
		scanf("%s", word);
		for (int i = 0; word[i] != '\0'; i++) {
			if (word[i] != word[i + 1]) {
				for (int j = i; word[j] != '\0'; j++) {
					if (word[i] == word[j + 1]) {
						cnt++;
						break;
					}
				}
				if (mcnt == cnt) {
					continue;
				}
			}
			if (word[i] == word[i + 1]) {
				continue;
			}
			else 
				break;
		}
	}
	printf("%d", num-cnt);
}
```
