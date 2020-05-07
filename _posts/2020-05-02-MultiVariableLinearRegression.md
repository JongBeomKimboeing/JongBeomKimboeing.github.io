```python
import tensorflow as tf
import numpy as np

data = np.array([[73., 80., 75., 152.],
                [93., 88., 93., 185.],
                [89., 91., 90., 180.],
                [96., 98., 100., 196.],
                [73., 66., 70., 142.]
                 ], dtype=np.float32)

x = data[:, :-1] #[행,열] ->5행 3열
y = data[:, [-1]]#5행 1열

w = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.0000001

def predict(X):
    return tf.matmul(X,w)+b

#gradient descent
n_epoch = 2000
for i in range(n_epoch+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(x)-y))

    w_grad, b_grad = tape.gradient(cost, [w,b]) # cost를 w1,w2,...,wn들의 wieght에 대해서 미분한다.
                                                # cost를 b1,b2,...,bn들의 bias에 대해서 미분한다.
    w.assign_sub(learning_rate*w_grad) # w1,w2,...,wn개의 weight들을 하나하나 씩 업데이트 시킨다.
    b.assign_sub(learning_rate*b_grad)# b1,b2,...,bn개의 bias들을 하나하나 씩 업데이트 시킨다.

    if i%100 == 0:
        print("%d | %f"%(i, cost.numpy()))
```
