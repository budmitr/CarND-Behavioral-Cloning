# CarND-Behavioral-Cloning

This is my implementation of P3 for [Udacity's Self Driving Car Engineer Nanodegree](https://www.udacity.com/drive)

# Preface

While working on this project I faced several environmental issues, especially network speed.
Since it was hard to transfer weights and datasets to and from my remote server with GPU, I focused on solution with these extra
requirements:
* stay within udacity dataset
* find model with less params to keep weights file small

# Data description

I used dataset provided by udacity as a training set.
For validation I recorded my own data on first track by passing one single lap.
Test data for final model evaluation is recorded by me on 2nd track by passing the road until deadend.

Training data is very unbalanced -- from 8000+ rows more than half is about straight driving, where steering angle is 0.
To solve this, I keep only 10% of 0-angle data:

```
train_source = pd.read_csv(DATA_TRAIN_FOLDER + 'driving_log.csv')
train_nonzero = train_source[train_source.steering != 0]
train_zero = (train_source[train_source.steering == 0]).sample(frac=.1)
data_train = pd.concat([train_nonzero, train_zero], ignore_index=True)
```

Resulting dataset is just 4111 rows.
Moreover, I will not use right\left cams (but I tried them also), so 4111 images is my full training data.
Pretty small yeah :)
Here is how the balanced data histogram looks like.

![Train data histogram](readme_images/data_train_hist.png "Train data histogram")

# Data augmentation

To train a model with such small data, I have to use augmentation.
First idea is to flip images horizontally and negate the steering angle:
```
def _get_flipped_image(img, y):
    img = cv2.flip(img, 1)
    return img, -y
```

Inspired by [Vivek Yadav's post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
I used brightness and translation augmentations.
Noteworthy is that even with ready code I had to play with params for hours to adjust them for my case.
```
def _get_brightnessed_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    img[:,:,2] = img[:,:,2] * random_bright
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def _get_translated_image(img, y, trans_range):
    rows, cols, _ = img.shape
    tr_x = trans_range * np.random.uniform() - trans_range/2
    y = y + tr_x/trans_range * 2 *.4
    tr_y = 10 * np.random.uniform() - 10/2
    Trans_M = np.float32([[1,0, tr_x], [0,1, tr_y]])
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    return img, y
```

That's it for my augmentation. I used 50% chance for images to be flipped:
```
def image_augmentation(img, y):
    if np.random.uniform() < 0.5:
        img, y = _get_flipped_image(img, y)
    img = _get_brightnessed_image(img)
    img, y = _get_translated_image(img, y, 100)
    return img, y
```

Another idea was to cut useless part of images.
This preprocessing is used for train and prediction code aswell:
```
def image_transformation(img):
    img = img[60:-20,:,:]
    img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
    return img
```

# Data generators

I have 3 generators: one for train, one for validation and one for test sets.
Train generator has to make random data.

Validation and test generators sequentially go through the data to ensure repeatability and consistency of evaluations.
Here is how all of them loos like:

Train data generator
```
def train_data_generator(batch_size):
    while True:
        X = np.zeros((batch_size, *input_shape), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.float32)
        for idx in range(batch_size):
            row = data_train.sample()
            img, steering = _get_img_and_steering_from_row(row, DATA_TRAIN_FOLDER)
            img, steering = image_augmentation(img, steering)
            img = image_transformation(img)
            X[idx], y[idx] = img, steering
        yield X, y
```

Examples of train data images:
![Train data examples](readme_images/training_data_examples.png "Train data examples")

Validation data generator
```
def val_data_generator(batch_size):
    seq_idx = 0
    while True:
        X = np.zeros((batch_size, *input_shape), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.float32)
        for idx in range(batch_size):
            row = data_val.iloc[seq_idx]
            img, steering = _get_img_and_steering_from_iloc_row(row, DATA_VAL_FOLDER)
            img = image_transformation(img)
            X[idx], y[idx] = img, steering
            
            seq_idx += 1
            if seq_idx == len(data_val):
                seq_idx = 0
        yield X, y
```
Examples of validation data images:
![Validation data examples](readme_images/validation_data_examples.png "Validation data examples")


Test data generator
```
def test_data_generator(batch_size):
    seq_idx = 0
    while True:
        X = np.zeros((batch_size, *input_shape), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.float32)
        for idx in range(batch_size):
            row = data_test.iloc[seq_idx]
            img, steering = _get_img_and_steering_from_iloc_row(row, DATA_TEST_FOLDER)
            img = image_transformation(img)
            X[idx], y[idx] = img, steering
            
            seq_idx += 1
            if seq_idx == len(data_test):
                seq_idx = 0
        yield X, y
```

Examples of test data images:
![Test data examples](readme_images/test_data_examples.png "Test data examples")

# Model

I tried several different models: [NVIDIA's](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf),
[commaai's](https://github.com/commaai/research/blob/master/train_steering_model.py),
[VGG](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) and several own ones.

After many hours of searches and fights with size constraints, I used model by [Vivek Yadav](https://github.com/vxy10/P3-BehaviorCloning) and it gave me best results

Here is original visualization of the model [from original post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.bslwaytx4)
![Model architecture](https://cdn-images-1.medium.com/max/1600/1*47fIMy2fL2lc6Q1drpyYvQ.png)

# Training process

With baches of size 50 trained the model for 10 epoch with 20k examples in each.
Validation data was used to evaluate model performance.
I used [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) callback of keras to save best weights according to validation data evaluation.

```
model = get_vivek_model()
model.fit_generator(
    train_data_generator(50),
    samples_per_epoch=20000,
    nb_epoch=10,
    validation_data=val_data_generator(250),
    nb_val_samples=750,
    callbacks=[ModelCheckpoint(filepath="best_validation.h5", verbose=1, save_best_only=True)]
)
```

# Evaluation

After model was trained, I print 3 evaluation metrics:
* latest epoch weights on val data: 0.00808320178961
* best weights on val data: 0.00384507623191
* final evaluation on test data: 0.0083494499007141908

Manual checks of how the model predicts the output on **validation** data:
```
0.0 [-0.01915371]
-0.02204349 [-0.07885508]
-0.1382363 [-0.13422723]
0.0 [-0.03817816]
0.0 [-0.03485824]
0.0 [ 0.00770417]
-0.1866501 [-0.12932573]
0.0 [ 0.00884948]
0.0 [-0.0196645]
-0.07045718 [-0.04182312]
```

Manual checks of how the model predicts the output on **test** data:
```
-0.2389614 [-0.23317344]
0.0 [-0.16454121]
0.0 [-0.03975093]
0.0 [-0.06640729]
0.0 [ 0.03869127]
0.007028675 [ 0.00721287]
0.0 [ 0.02606358]
0.0 [ 0.02242848]
-0.2872807 [-0.3301293]
-0.1713143 [-0.16757624]
```

This is actually pretty good.
In worst cases I had something like predicting positive angle by having negative one with big deviations :)

# Save and use

Saving code is very straightforward:
```
import json
with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('model.h5')
```

In `drive.py` I added image transformation code only (same as for generators):
```
def preprocess_image(img):
    img = img[60:-20,:,:]
    img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
    return img
```

# Acknowledgements

Big thanks to [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) 
for sharing his approach in augmentation and model architecture
