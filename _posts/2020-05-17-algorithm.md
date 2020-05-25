# 프린터큐
정말 힘들게 풀어냈다... 이 문제는 정말 복합적인 문제였다. 큐와 우선순위 큐를 동시에 이용해야했다.
이번 문제에서 처음으로 STL을 써 보았는데, 확실히 STL이 편하다는 것을 느꼈다.
(차마 우선순위 큐를 구현해 놨는데, 큐를 구현 한 걸 쓰기엔 너무 힘들었다.)
이번 문제를 통해 자료구조를 어떻게 응용하는 지 알게 됐다.

```c
#include<stdio.h>
#include <queue>
using namespace std;
queue <pair<int, int>> q;
#define HEAP_LEN 102

typedef struct heapElem {
	int priority;
	int data;
}HeapElem;

typedef struct heap {
	int numofData;
	HeapElem heapArr[HEAP_LEN];
}Heap;

void heapInit(Heap* myheap) {
	myheap->numofData = 0;
}

int HIsEmpty(Heap* myheap) {
	if (myheap->numofData == 0) {
		return 1;
	}
	else
		return 0;
}

int GetParentIDX(int idx) {
	return idx / 2;
}

int GetLChildIDX(int idx) {
	return idx * 2;
}

int GetRChildIDX(int idx) {
	return idx * 2 + 1;
}

int GetHiChildPriorityIDX(Heap* myheap, int idx) {
	if (GetLChildIDX(idx) > myheap->numofData) {
		return 0;
	}
	else if (GetLChildIDX(idx) == myheap->numofData) {
		return GetLChildIDX(idx);
	}
	else
	{
		if (myheap->heapArr[GetLChildIDX(idx)].priority < myheap->heapArr[GetRChildIDX(idx)].priority) {
			return GetRChildIDX(idx);
		}
		else
			return GetLChildIDX(idx);
	}

}

void HInsert(Heap* myheap, int data, int priority) {
	HeapElem nelem = { priority, data };
	int i = myheap->numofData + 1;
	while (i != 1) {
		if (priority > myheap->heapArr[GetParentIDX(i)].priority) {
			myheap->heapArr[i] = myheap->heapArr[GetParentIDX(i)];
			i = GetParentIDX(i);
		}
		else
			break;
	}
	myheap->heapArr[i] = nelem;
	myheap->numofData++;
}


int HDelete(Heap* myheap) {
	int retdata = (myheap->heapArr[1]).data;
	HeapElem lastElem = myheap->heapArr[myheap->numofData];

	int parentidx = 1;
	int childidx;

	while (childidx = GetHiChildPriorityIDX(myheap, parentidx)) {
		if (lastElem.priority >= myheap->heapArr[childidx].priority)
			break;

		myheap->heapArr[parentidx] = myheap->heapArr[childidx];
		parentidx = childidx;
	}
	myheap->heapArr[parentidx] = lastElem;
	myheap->numofData--;
	return retdata;
}

int main(void) {
	Heap heap;
	int num = 0;

	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		int cnt = 0;
		heapInit(&heap);
		int printnum = 0;
		int question = 0;
		int importance = 0;
		scanf("%d %d", &printnum, &question);
		for (int n = 0; n < printnum; n++) {
			scanf("%d", &importance);
			q.push({ n, importance });
			HInsert(&heap, 0, importance);
		}
		while (q.size()) {
			int idx = q.front().first;
			int imp = q.front().second;
			q.pop();
			if (imp == heap.heapArr[1].priority) {
				HDelete(&heap);
				cnt++;
				if (idx == question) {
					printf("%d\n", cnt);
					break;
				}
			}
			else {
				q.push({ idx, imp });
			}
		}
		while (!q.empty()) {
			q.pop();
		}

	}
	return 0;

}
```
