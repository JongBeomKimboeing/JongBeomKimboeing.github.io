# 이진트리 구현
```c
#include<stdio.h>
#include<stdlib.h>

typedef struct node {
	struct node* leftnode;
	int data;
	struct node* rightnode;
}Node;

Node* MakeBTreeNode(void){
	Node* newNode = (Node*)malloc(sizeof(Node));
	newNode->leftnode = NULL;
	newNode->rightnode = NULL;
	return newNode;
}

int GetData(Node* nd) {
	return nd->data;
}

void SetData(Node* nd, int data) {
	nd->data = data;
}

Node* GetLeftSubTree(Node* nd) {
	return nd->leftnode;
}

Node* GetRightSubTree(Node* nd) {
	return nd->rightnode;
}

void MakeLeftSubTree(Node* main, Node* sub) {
	if (main->leftnode != NULL) {
		free(main->leftnode);
	}
		main->leftnode = sub;
}

void MakeRightSubTree(Node* main, Node* sub) {
	if (main->rightnode != NULL) {
		free(main->rightnode);
	}
	main->rightnode = sub;
}

int main(void) {
	Node* node1 = MakeBTreeNode();
	Node* node2 = MakeBTreeNode();
	Node* node3 = MakeBTreeNode();
	Node* node4 = MakeBTreeNode();

	SetData(node1, 1);
	SetData(node2, 2);
	SetData(node3, 3);
	SetData(node4, 4);

	MakeLeftSubTree(node1, node2);
	MakeRightSubTree(node1, node3);
	MakeLeftSubTree(node2, node4);

	printf("%d\n", GetData(GetLeftSubTree(node1)));
	printf("%d\n", GetData(GetLeftSubTree(GetLeftSubTree(node1))));

	return 0;
}
```
