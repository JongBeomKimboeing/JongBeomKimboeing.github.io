
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
