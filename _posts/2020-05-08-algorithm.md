---
layout: post
title: 5월 8일 문제풀이
description: "5월 8일 문제풀이"
modified: 2020-05-08
tags: [알고리즘,자료구조기초]
categories: [백준문제풀이]
---
# 5월 6일(2문제) (10828번, 9012번)
<br>
## 10828번 (스택)
```c
#include<stdio.h>
#include<string.h>

typedef struct STACK {
	int stack[1000];
	int curpoint;
}Stack;

void Push(Stack *stacks, int pushint) {
	stacks->stack[stacks->curpoint] = pushint;
	stacks->curpoint++;
}

int Pop(Stack *stacks) {
	int temp;
	if (stacks->curpoint == 0) {
		return -1;
	}
	stacks->curpoint--;
	temp = stacks->stack[stacks->curpoint];
	stacks->stack[stacks->curpoint] = 0;
	return temp;
}
int size(Stack* stacks) {
	if (stacks->curpoint == 0) {
		return 0;
	}
	return stacks->curpoint;
}

int empty(Stack* stacks) {
	if (stacks->curpoint == 0) {
		return 1;
	}
	else
		return 0;
}

int top(Stack* stacks) {
	if (stacks->curpoint == 0)
		return -1;
	else {
		return stacks->stack[stacks->curpoint - 1];
	}
}

int main(void) {
	Stack stack;
	stack.curpoint = 0;
	char command[10];
	int num = 0;
	int pushnum = 0;
	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		char command[10] = {'\0'};
		scanf("%s",command);
		if (!strcmp(command, "push")){
			int nnum = 0;
			scanf("%d", &nnum);
			Push(&stack, nnum);
		}
		if (!strcmp(command, "top")) {
			printf("%d\n", top(&stack));
		}
		if (!strcmp(command, "size")) {
			printf("%d\n", size(&stack));
		}
		if (!strcmp(command, "empty")) {
			printf("%d\n", empty(&stack));
		}
		if (!strcmp(command, "pop")) {
			printf("%d\n", Pop(&stack));
		}
	}
}
```
<br>
## 9012번 (괄호)
스택을 이용한 문
```c
#include<stdio.h>
#include<string.h>

typedef struct STACK {
	char stack[1000];
	int curpoint;
}Stack;

void push(Stack *stacks, char pushchar) {
	stacks->stack[stacks->curpoint] = pushchar;
	stacks->curpoint++;
}

char pop(Stack *stacks) {
	char temp;
	if (stacks->curpoint == 0) {
		return '\0';
	}
	stacks->curpoint--;
	temp = stacks->stack[stacks->curpoint];
	stacks->stack[stacks->curpoint] = 0;
	return temp;
}
int size(Stack* stacks) {
	if (stacks->curpoint == 0) {
		return 0;
	}
	return stacks->curpoint;
}

int empty(Stack* stacks) {
	if (stacks->curpoint == 0) {
		return 1;
	}
	else
		return 0;
}

char top(Stack* stacks) {
	if (stacks->curpoint == 0)
		return 'r';
	else {
		return stacks->stack[stacks->curpoint - 1];
	}
}

void initstack(Stack* stacks) {
	stacks->curpoint = 0;
	char stack[1000] = {'\0'};
}

int main(void) {
	Stack stack;
	stack.curpoint = 0;
	char vps[52];
	int num = 0;
	int now = 0;
	int pushnum = 0;
	int j = 0;
	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		char vps[52] = { '\0' };
		initstack(&stack);
		scanf("%s", vps);
		for (j = 0; vps[j] != '\0'; j++) {
			if (vps[j] == '(') {
				push(&stack, vps[j]);
			}
			if (vps[j] == ')') {
				if (empty(&stack)) {
					push(&stack, vps[j]);
					break;
				}
				else
					pop(&stack);
			}
		}
		if (empty(&stack)) {
			now = 1;
		}

		if (!empty(&stack)) {
			now = 0;
		}

		if (now == 1) {
			printf("YES\n");
		}
		else if (now == 0) {
			printf("NO\n");
		}
	}
}
```
