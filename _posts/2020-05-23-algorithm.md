---
layout: post
title: 5월 23일 문제풀이(덱)
description: "5월 23일 자료구조 연습"
modified: 2020-05-23
tags: [알고리즘,자료구조기초]
categories: [백준문제풀이]
---

# 덱 구현 (10866번)

그동안 수술떄문에 알고리즘 문제를 못 풀었다.<br>
단순 덱 구현문제라 가벼운 마음으로 재밌게 풀었다.<br>
pop_front함수와 pop_back함수에서 덱이 empty될 때의 경우를 안 넣어줘서 오답처리를 받았었다.<br>


```c
#include<stdio.h>
#include<stdlib.h>
typedef struct node {
	struct node* frontnode;
	int data;
	struct node* backnode;
}Node;

typedef struct deque {
	Node* front;
	Node* rear;
	Node* peak;
	int numOfData;
}Deque;

void DequeInit(Deque* deque) {
	deque->front = NULL;
	deque->rear = NULL;
	deque->numOfData = 0;
}

void push_front(Deque* deque, int data) {
	Node* newnode = (Node*)malloc(sizeof(Node));
	if (empty(deque)) {
		deque->front = newnode;
		deque->rear = newnode;
		newnode->frontnode = NULL;
		newnode->backnode = NULL;
		newnode->data = data;
		deque->numOfData++;
	}
	else {
		newnode->backnode = deque->front;
		deque->front->frontnode = newnode;
		deque->front = newnode;
		newnode->frontnode = NULL;
		deque->front->data = data;
		deque->numOfData++;
	}
}

void push_back(Deque* deque, int data) {
	Node* newnode = (Node*)malloc(sizeof(Node));
	if (empty(deque)) {
		deque->front = newnode;
		deque->rear = newnode;
		newnode->frontnode = NULL;
		newnode->backnode = NULL;
		newnode->data = data;
		deque->numOfData++;
	}
	else {
		newnode->frontnode = deque->rear;
		deque->rear->backnode = newnode;
		deque->rear = newnode;
		newnode->backnode = NULL;
		deque->rear->data = data;
		deque->numOfData++;
	}
}

int pop_front(Deque* deque) {
	if (empty(deque)) {
		return -1;
	}
	else {
		int tempdata = deque->front->data;
		Deque* tempad = deque->front;
		deque->front = deque->front->backnode;
		free(tempad);
		deque->numOfData--;
		if (deque->front == NULL) {  //이 부분을 생각을 못 해내서 백준에서 정답처리를 못 받았었다.
			deque->rear = NULL;
		}
		else {
			deque->front->frontnode = NULL;
		}
		return tempdata;
	}
}

int pop_back(Deque* deque) {
	if (empty(deque)) {
		return -1;
	}
	else {
		int tempdata = deque->rear->data;
		Deque* tempad = deque->rear;
		deque->rear = deque->rear->frontnode;
		free(tempad);
		deque->numOfData--;
		if (deque->rear == NULL) { //이 부분을 생각을 못 해내서 백준에서 정답처리를 못 받았었다.
			deque->front = NULL;
		}
		else {
			deque->rear->backnode = NULL;
		}
		return tempdata;
	}
}

int front(Deque* deque) {
	if (deque->front == NULL) {
		return -1;
	}
	else {
		return deque->front->data;
	}
}

int back(Deque* deque) {
	if (deque->rear == NULL) {
		return -1;
	}
	else {
		return deque->rear->data;
	}
}

int empty(Deque* deque) {
	if (deque->numOfData == 0) {
		return 1;
	}
	else
		return 0;
}

int size(Deque* deque) {
	return deque->numOfData;
}

int main(void) {
	Deque myDeque;
	DequeInit(&myDeque);
	int num = 0;
	char command[15];
	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		char command[15] = { '\0' };
		scanf("%s", command);
		if (!strcmp(command, "push_front")) {
			int input = 0;
			scanf("%d", &input);
			push_front(&myDeque, input);
		}
		if (!strcmp(command, "push_back")) {
			int input = 0;
			scanf("%d", &input);
			push_back(&myDeque, input);
		}
		if (!strcmp(command, "pop_front")) {
			printf("%d\n", pop_front(&myDeque));
		}
		if (!strcmp(command, "pop_back")) {
			printf("%d\n", pop_back(&myDeque));
		}
		if (!strcmp(command, "size")) {
			printf("%d\n", size(&myDeque));
		}
		if (!strcmp(command, "empty")) {
			printf("%d\n", empty(&myDeque));
		}
		if (!strcmp(command, "front")) {
			printf("%d\n", front(&myDeque));
		}
		if (!strcmp(command, "back")) {
			printf("%d\n", back(&myDeque));
		}

	}
}

```
