---
layout: post
title: 문제2675번(문자열 반복)
description: "문제2675번"
modified: 2020-05-02
tags: [알고리즘,배열]
categories: [백준문제풀이]
---
# 문제가 그나마 쉬워서 그방 풀었다.
```c
#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>

int main(void) {
	int j = 0, m = 0;
	char s[20] = {'\0'};
	scanf("%d", &m);
	for (int n = 0; n < m; n++) {
		int i = 0;
		scanf("%d %s", &j, s);
		while (s[i] != '\0') {
			for (int l = 0; l < j; l++) {
				printf("%c", s[i]);
			}
			i++;
		}
		printf("\n");
	}
}
```
