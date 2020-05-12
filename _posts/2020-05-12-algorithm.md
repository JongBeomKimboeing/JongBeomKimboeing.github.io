```c
#include<stdio.h>
#include<stdlib.h>

typedef struct node {
	int data;
	struct node* nextnode;
}Node;

typedef struct list {
	Node* head;
	Node* cur;
	Node* before;
	int numOfData;
}List;

void ListInit(List* plist) {
	plist->head = (Node*) malloc(sizeof(Node)); 
	//dummy node를 구조체로 생성 (head가 Node*형이므로 맞춰서 Node*형으로 변환시킨다.)
	//plist->head에 Node의 주소를 넣어주기 위해 (Node*)형으로 형변환 시켜준다.
	//(Node*)형으로 형변환을 시키면 Node 구조체의 주소가 전달됨. 
	//그 의미는 생성된 구조체의 주소를 전달해 줬다는 것과 의미가 같음.
	plist->head->nextnode = NULL;
	plist->numOfData = 0;
}

void LInsert(List* plist, int data) {
	Node* newNode = (Node*)malloc(sizeof(Node));
	newNode->nextnode = plist->head->nextnode;
	plist->head->nextnode = newNode;
	newNode->data = data;
	plist->numOfData++;
}

int LFirst(List* plist, int* pdata) {
	if (plist->head->nextnode == NULL) {
		return 0;
	}
	else {
		plist->before = plist->head;
		plist->cur = plist->head->nextnode;
		*pdata = plist->cur->data;
		return 1;
	}
}

int LNext(List* plist, int* pdata) {
	if (plist->cur->nextnode == NULL) {
		return 0;
	}
	else {
		plist->cur = plist->cur->nextnode;
		plist->before = plist->before->nextnode;
		*pdata = plist->cur->data;
		return 1;
	}
}

int LRemove(List* plist) {
	Node* tempNode = plist->cur;
	int temp = plist->cur->data;
	plist->before->nextnode = plist->cur->nextnode;
	plist->cur = plist->before;
	free(tempNode);
	(plist->numOfData)--;
	return temp;
}

int LCount(List* plist) {
	return plist->numOfData;
}



int main(void) {
	List list;
	int data = 0;
	ListInit(&list);

	LInsert(&list, 11);
	LInsert(&list, 22);
	LInsert(&list, 33);
	printf("현재 데이터 수: %d\n", LCount(&list));

	if (LFirst(&list, &data)) {
		if (data == 22) {
			LRemove(&list);
		}
		while (LNext(&list, &data)) {
			if (data == 22) {
				LRemove(&list);
			}
		}
	}
	printf("현재 데이터 수: %d\n", LCount(&list));

	if (LFirst(&list, &data)) {
		printf("%d", data);
		while (LNext(&list, &data)) {
			printf("%d", data);
		}
	}
	
}
```
