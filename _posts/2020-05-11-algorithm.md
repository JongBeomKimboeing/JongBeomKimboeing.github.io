
```c
#define _CRT_SECURE_NO_WARNINGS
#define QLEN 10001
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

typedef struct que {
	int front;
	int rear;
	int que_array[QLEN];
}Que;

int push(Que* que,int input) {
	/*
	if ((abs(que->rear - que->front)==1 && size(que)== QLEN)) {
		return 0;
	}
	if (que->rear == QLEN && size(que) == QLEN) {
		return 0;
	}*/
	if (que->rear == QLEN) {
		que->rear = 0;
	}
	que->rear++;
	que->que_array[que->rear] = input;
	return que->que_array[que->rear];
}


int pop(Que* que) {
	int temp = 0;
	if (empty(que)) {
		return -1;
	}
	if (que->front == QLEN) {
		que->front = 0;
	}
	
	que->front++;
	temp = que->que_array[que->front];
	que->que_array[que->front] = 0;
	return temp;
}
int empty(Que* que) {
	if (que->front == que->rear) {
		return 1;
	}
	else
		return 0;
}

int front(Que* que) {
	if (empty(que)) {
		return -1;
	}
	else
		return que->que_array[que->front+1];
}

int back(Que* que) {
	if (empty(que)) {
		return -1;
	}
	else
		return que->que_array[que->rear];
}

int size(Que* que) {
	return que->rear - que->front;
}


int main(void) {
	Que que = { 0,0,{0} };
	int num = 0;
	char command[7] ;
	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		char command[7] = { '\0' };
		scanf("%s", command);
		if (!strcmp(command, "push")) {
			int input = 0;
			scanf("%d", &input);
			push(&que, input);
		}
		if (!strcmp(command, "pop")) {
			printf("%d\n", pop(&que));
		}
		if (!strcmp(command, "size")) {
			printf("%d\n", size(&que));
		}
		if (!strcmp(command, "empty")) {
			printf("%d\n", empty(&que));
		}
		if (!strcmp(command, "front")) {
			printf("%d\n", front(&que));
		}
		if (!strcmp(command, "back")) {
			printf("%d\n", back(&que));
		}

	}
}
```
