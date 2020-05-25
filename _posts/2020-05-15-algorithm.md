---
layout: post
title: 5월 15일 문제풀이(트리순회)
description: "5월 15일 자료구조 연습"
modified: 2020-05-15
tags: [알고리즘,자료구조기초]
categories: [백준문제풀이]
---
# 트리 순회(1991번)
단순 트리 순회를 만드는 건 어렵지 않았지만, 문제에서 제시된 방법으로 입력을 받고, 출력을 받는 구현에 있어서 어려움을 겪었다.<br>
이번 문제를 통해 문자 데이터를 잘 활용해야겠다는 생각을 했다.

```c
#include<stdio.h>
#include<stdlib.h>

typedef struct node {
	struct node* leftnode;
	char data;
	struct node* rightnode;
}Node;

Node* MakeBTreeNode(char data){
	Node* newNode = (Node*)malloc(sizeof(Node));
	newNode->leftnode = NULL;
	newNode->rightnode = NULL;
	newNode->data = data;
	return newNode;
}

char GetData(Node* nd) {
	return nd->data;
}

void SetData(Node* nd, char data) {
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

void Inorder_Traversal(Node* root) {
	if (root == NULL) {
		return;
	}
	Inorder_Traversal(root->leftnode);
	printf("%c", root->data);
	Inorder_Traversal(root->rightnode);
}

void Preorder_Traversal(Node* root) {
	if (root == NULL) {
		return;
	}
	printf("%c", root->data);
	Preorder_Traversal(root->leftnode);
	Preorder_Traversal(root->rightnode);
}

void Postorder_Traversal(Node* root) {
	if (root == NULL) {
		return;
	}
	Postorder_Traversal(root->leftnode);
	Postorder_Traversal(root->rightnode);
	printf("%c", root->data);
	
}



int main(void) {
	int num = 0;
	char first = '\0';
	Node *node[28];
	for (int i = 0; i < 27; i++) {
		node[i] = MakeBTreeNode('A'+i);
	}
	scanf("%d\n", &num);
	for (int i = 0; i < num; i++) {
		char a, b, c ='\0';
		scanf("%c %c %c", &a, &b, &c);
		getchar();
		if (i == 0) {
			first = a;
		}
		if (b == '.' && c == '.') {
			MakeLeftSubTree(node[a - 'A'], NULL);
			MakeRightSubTree(node[a - 'A'], NULL);
		}
		else if (b == '.') {
			MakeLeftSubTree(node[a - 'A'], NULL);
			MakeRightSubTree(node[a - 'A'], node[c - 'A']);
		}
		else if (c == '.') {
			MakeLeftSubTree(node[a - 'A'], node[b - 'A']);
			MakeRightSubTree(node[a - 'A'], NULL);
		}
		else {
			MakeLeftSubTree(node[a - 'A'], node[b - 'A']);
			MakeRightSubTree(node[a - 'A'], node[c - 'A']);
		}
	}
	Preorder_Traversal(node[first - 'A']);
	printf("\n");
	Inorder_Traversal(node[first-'A']);
	printf("\n");
	Postorder_Traversal(node[first - 'A']);
}
```

