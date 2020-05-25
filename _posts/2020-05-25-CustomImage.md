---
layout: post
title: Custom Image(내가 가지고 있는 데이터)를 사용하여 Image 분류하기
description: "Image classification with custom image(my data)"
modified: 2020-05-25
tags: [김성훈,DL]
categories: [김성훈DL]
---

# Custom Image를 이용하여 Image Classification 해보기
항상 MNIST와 같은 만들어진 데이터셋에대한 훈련만 하다보니<br>
뭔가 나만의 데이터를 이용하여 Image Classification을 해보고 싶었다.<br>
그렇다고 고양이와 개 분류 같은 사람들이 많이 해 본 분류는 안 해보고 싶어서<br>
나와 아빠의 사진을 분류해보는 재밌는 실습을 해 보았다.

참고자료1: https://www.tensorflow.org/tutorials/images/classification<br>
참고자료2: https://www.tensorflow.org/tutorials/load_data/images

위 tensorflow tutorials에서 많은 참고를 했다.<br>
(위 사이트를 참고하면 아주 많은 도움이 된다.)

## 파일 만들기

├─image<br>
│  ├─train<br>
│  │  ├─father<br>
│  │  └─me<br>
│  └─validation<br>
│      ├─father<br>
│      └─me<br>

위와 같은 구조로 pycharm project에 직접 image파일을 만들었다.

## 이미지 넣기
father파일과 me파일에 직접 찍은 내 사진과 아빠 사진을 파일에 넣었다.

## 코드로 이미지파일 불러오기(customimage.py)
아래 코드는 파일을 불러오는 역할을 하는 코드이다.

```python
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator ,array_to_img, img_to_array, load_img

train_dir = os.path.join('/Users/harry/PycharmProjects/20200515AI/image', 'train')
validation_dir = os.path.join('/Users/harry/PycharmProjects/20200515AI/image', 'validation')

train_father_dir = os.path.join(train_dir, 'father')  # directory with our training fahter pictures
train_me_dir = os.path.join(train_dir, 'me')  # directory with our training me pictures
validation_father_dir = os.path.join(validation_dir, 'father')  # directory with our validation father pictures
validation_me_dir = os.path.join(validation_dir, 'me')

num_father_tr = len(os.listdir(train_father_dir))
num_me_tr = len(os.listdir(train_me_dir))

num_father_val = len(os.listdir(validation_father_dir))
num_me_val = len(os.listdir(validation_me_dir))

total_train = num_father_tr + num_me_tr
total_val = num_father_val + num_me_val

print('total training father images:', num_father_tr)
print('total training me images:', num_me_tr)

print('total validation father images:', num_father_val)
print('total validation me images:', num_me_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
```
## 이미지 불리기
사실 직접 찍은 사진은 아빠와 나의 사진 각각 40장 정도로 총 80장만 찍었다.<br>
그런데, 실제로 훈련을 시킬 떄 훈련데이터를 6000개로 훈련했다.<br>
데이터를 변형시켜 훈련데이터를 더 많이 만들어서 80장을 6000장으로 만들 수 있었다.<br>

## 변형된 이미지 예시
![image](/assets/gDgraph.png)
![image](/assets/gDgraph.png)

### 이미지 불리기 코드 (increase_data.py)
사실 아래코드는 완벽한 코드는 아니라고 생각한다.<br>
일일이 변형된 이미지를 만들 이미지파일을 선택해 주어야하기 때문이다.<br>
다음에 시간이 되면 더 효율적인 코드를 만들어보고자 한다.<br>
변형된 이미지는 반드시 training할 이미지파일에 넣어야한다.<br>
(즉, 변형된 이미지를 test이미지로 설정해주면 안 된다!)<br>
추가적으로, 이미지 불리기 코드는 학습하는 코드와 분리시켜 만들었다.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator ,array_to_img, img_to_array, load_img
import os

#파일에 있는 이미지 현황 출력----------------------------------------------------------
train_dir = os.path.join('/Users/harry/PycharmProjects/20200515AI/image', 'train') 
validation_dir = os.path.join('/Users/harry/PycharmProjects/20200515AI/image', 'validation')

train_father_dir = os.path.join(train_dir, 'father')  # directory with our training cat pictures
train_me_dir = os.path.join(train_dir, 'me')  # directory with our training dog pictures
validation_father_dir = os.path.join(validation_dir, 'father')  # directory with our validation cat pictures
validation_me_dir = os.path.join(validation_dir, 'me')

num_father_tr = len(os.listdir(train_father_dir))
num_me_tr = len(os.listdir(train_me_dir))

num_father_val = len(os.listdir(validation_father_dir))
num_me_val = len(os.listdir(validation_me_dir))

total_train = num_father_tr + num_me_tr
total_val = num_father_val + num_me_val

print('total training father images:', num_father_tr)
print('total training me images:', num_me_tr)

print('total validation father images:', num_father_val)
print('total validation me images:', num_me_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
#------------------------------------------------------------------

#이미지 변형시킬 image generator 설정
train_image_generator = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=.15, height_shift_range=.15, horizontal_flip=True, zoom_range=0.5)

#이미지를 변형시키는 함수
def more_image(image_address, savefile,name):
    img = load_img(image_address)
    x = img_to_array(img) #(961, 721, 3)
    x = x.reshape((1,) + x.shape) #차원수를 하나 높여줌 (961, 721, 3)
    i = 0
    # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
    for batch in train_image_generator.flow(x, batch_size=1, save_to_dir=savefile, save_prefix=name, save_format='jpg'):
        i += 1
        if i > 1:
            break

#변형된 이미지를 만들 이미지를 설정
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/me/KakaoTalk_20200515_185332208_15.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/me','newme')
'''
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/father/KakaoTalk_20200515_191341327_09.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/father','newfather')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/me/KakaoTalk_20200515_185332208_16.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/me','newme')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/father/KakaoTalk_20200515_191341327_10.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/father','newfather')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/me/KakaoTalk_20200515_185332208_17.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/me','newme')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/father/KakaoTalk_20200515_191555562.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/father','newfather')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/me/KakaoTalk_20200515_185332208_18.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/me','newme')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/father/KakaoTalk_20200515_191555562_01.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/father','newfather')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/me/KakaoTalk_20200515_191653547.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/me','newme')
more_image('/Users/harry/PycharmProjects/20200515AI/image/train/father/KakaoTalk_20200515_191555562_02.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/father','newfather')
'''

#파일에 있는 이미지 현황 출력----------------------------------------------------------
num_father_tr = len(os.listdir(train_father_dir))
num_me_tr = len(os.listdir(train_me_dir))

num_father_val = len(os.listdir(validation_father_dir))
num_me_val = len(os.listdir(validation_me_dir))

total_train = num_father_tr + num_me_tr
total_val = num_father_val + num_me_val

print('total training father images:', num_father_tr)
print('total training me images:', num_me_tr)

print('total validation father images:', num_father_val)
print('total validation me images:', num_me_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
#-------------------------------------------------------------------------------------------
```
## 이미지 불러오기와 학습 코드 전체 (customimage.py)
```python
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator ,array_to_img, img_to_array, load_img

#이미지 데이터 불러오기-----------------------------------------------------------------
train_dir = os.path.join('/Users/harry/PycharmProjects/20200515AI/image', 'train')
validation_dir = os.path.join('/Users/harry/PycharmProjects/20200515AI/image', 'validation')

train_father_dir = os.path.join(train_dir, 'father')  # directory with our training cat pictures
train_me_dir = os.path.join(train_dir, 'me')  # directory with our training dog pictures
validation_father_dir = os.path.join(validation_dir, 'father')  # directory with our validation cat pictures
validation_me_dir = os.path.join(validation_dir, 'me')

num_father_tr = len(os.listdir(train_father_dir))
num_me_tr = len(os.listdir(train_me_dir))

num_father_val = len(os.listdir(validation_father_dir))
num_me_val = len(os.listdir(validation_me_dir))

total_train = num_father_tr + num_me_tr
total_val = num_father_val + num_me_val

print('total training father images:', num_father_tr)
print('total training me images:', num_me_tr)

print('total validation father images:', num_father_val)
print('total validation me images:', num_me_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
#-------------------------------------------------------------------

train_batch_size = 6756
test_batch_size = 30
batch_size = 100
training_epoch = 1
learning_rate = 0.0001
IMG_HEIGHT = 150
IMG_WIDTH = 150

Label = ['father', 'me']
#이미지 불리기(별개의 python파일을 만들어 사용하는 게 바람직하다.)--------------------------------------------------
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

def more_image(image_address, savefile,name):
    img = load_img(image_address)
    x = img_to_array(img) #(961, 721, 3)
    x = x.reshape((1,) + x.shape) #차원수를 하나 높여줌 (961, 721, 3)
    print(x.shape) #(1, 961, 721, 3)
    i = 0
    # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
    for batch in train_image_generator.flow(x, batch_size=1, save_to_dir=savefile, save_prefix=name, save_format='jpg'):
        i += 1
        if i > 50:
            break
            
#more_image('/Users/harry/PycharmProjects/20200515AI/image/train/me/KakaoTalk_20200515_173220100_24.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/me','newme')
#more_image('/Users/harry/PycharmProjects/20200515AI/image/train/father/KakaoTalk_20200515_173220100_13.jpg','/Users/harry/PycharmProjects/20200515AI/image/train/father','newfather')
#-----------------------------------------------------------------------------------------------------------------------
# train data와 test data 전처리(이미지를 labeling해주고 이미지 사이즈도 조정하는 등 train하기 위해 image를 전처리 해준다.) -------------
train_data_gen = train_image_generator.flow_from_directory(batch_size=train_batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=test_batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')
#------------------------------------------------------------------------------------------------------------------------------------

# 이미지가 잘 불러와 줬는지 테스트해보기 위해 출력해본다.---------------------------------------------------------
def show_batch(image_batch, label_batch):
    fig = plt.subplots(figsize=(20,20))
    for n in range(14):
        ax = plt.subplot(2,7,n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

x_train, y_train = next(train_data_gen)
x_test, y_test = next(val_data_gen)
#show_batch(x_train, y_train)
#--------------------------------------------------------------------------------------------------------

# 학습 model을 만들어준다 --------------------------------------------------------------------
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Conv2D(filters= 20, kernel_size=(25,25),kernel_initializer='he_uniform', padding="same", input_shape=(IMG_HEIGHT,IMG_WIDTH ,3), activation='relu'))
# tf.keras.layers.Conv2D-> filters= : fiter의 개수, kernel_size=: kernal 크기,  input_shape=: input data의 모양
#                       -> strides=(1, 1): filter를 얼마만큼의 stride로 움직일 것인가.
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(5,5)))

#layer2
tf.model.add(tf.keras.layers.Conv2D(filters= 30, kernel_size=(10,10),kernel_initializer='he_uniform', padding="same", activation='relu'))
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))

#layer3
tf.model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(20,20),kernel_initializer='he_uniform', strides=(2,2), padding="same",activation='relu'))
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#layer3(Fully connected)
tf.model.add(tf.keras.layers.Flatten())
tf.model.add(tf.keras.layers.BatchNormalization())
tf.model.add(tf.keras.layers.Dense(units=100,kernel_initializer='lecun_normal',activation='selu'))
#layer4(Fully connected)
tf.model.add(tf.keras.layers.BatchNormalization())
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=50,kernel_initializer='lecun_normal',activation='selu'))
#layer5(Fully connected)
tf.model.add(tf.keras.layers.BatchNormalization())
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=25,kernel_initializer='lecun_normal',activation='selu'))
#layer(Fully connected)
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=1,kernel_initializer='glorot_normal', activation='relu'))
tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
tf.model.summary()

models = []
num_models = 1
for m in range(num_models):
    models.append(tf.model)

predictions = np.zeros([y_test.shape[0], 1])

# 각각의 model로부터 나온 prediction을 합한다.

for m_idx, m in enumerate(models):
    m.fit(x_train, y_train,batch_size=batch_size,epochs=training_epoch)
    p = m.predict(x_test)
    predictions += p
    evaluation = m.evaluate(x_test, y_test)
    print(m_idx+1, 'loss: ', evaluation[0])
    print(m_idx+1, 'accuracy', evaluation[1])
print("\n")
# ensemble한 결과의 accuracy를 계산한다.
ensemble_correct_prediction = tf.equal(tf.cast(predictions>=1.0, tf.float32), y_test)
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
#print('Ensemble accuracy:', ensemble_accuracy.numpy())
print("\n")

for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", Label[int(y_test[random_index])],
          "predicted y: ", Label[int(tf.cast(predictions[random_index]>=1.0, tf.float32)[0])])

r = random.randint(0, x_test.shape[0]-1)
print("Label:", Label[int(y_test[r])])
print("predicted:", Label[int(tf.cast(predictions[r]>=1.0, tf.float32)[0])])
plt.imshow(x_test[r].reshape(150,150,3), interpolation='nearest')
plt.show()

evaluation = tf.model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
#-----------------------------------------------------------------------------------------

tf.model.save('/Users/harry/PycharmProjects/20200515AI/father_me_model') # 학습한 weight값을 저장해둔다
# weight값을 저장해둔 이유는 바로 찍은 사진을 바로 줘서 추론하도록 해보고 싶었으나, 또 공부할 게 많아서 다음에 해보려고 한다.
# 참고적으로, weight을 저장해서 전이학습도 가능하다.
```


