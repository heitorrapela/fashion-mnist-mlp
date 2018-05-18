# fashion-mnist-mlp

Version of MLP or what people call too DNN in Keras (not so deep, only 1 input layer, 3 hidden layers and 1 output layer). It is a fully connected neural network with architecture: 728-256-128-100-10. 

Since Fashion Mnist is a Mnist dataset like, the images are 28x28x1, so the first layer there are 28*28 = 728 neurons in the input layer and in the output layer there are 10 classes to classify.

 The dataset was split in train, validation and test, where validation is 0.2 of original dataset. Important to guide the training phase. 

## Best results with this MLP model

| Train Acc |  Valid Acc  |  Test Acc  |
| :---         |     :---:      |          ---: |
| 0.8913  | 0.8941 |  **0.8833**  |

ps: "blah blah blah, I got more in train or valid", this doesn't count. The train and valid acc can be larger if you don't try to control overfit, so the most important value of this table is the Test Acc. 

To prevent overfit it was added dropout between each layer in train phase with a value of 0.4. 
Others optim was tried but SGD was the best one, in my test :) 

**Hyperparameters:** 

lr=0.01, momentum=0.975, decay=2e-06, nesterov=True, epochs=100 and batch_size=100

## Why you don't put more neurons or layers?

The version with more one layer and less one layer dont't have too much impact in Test Acc and if you try to put more neurons you don't guarantee a better acc.  I was trying to train a model that has a balance of accuracy/memory, and this was a good one.

## Keras summary  (MLP Model):
 
| Layer (type)    |  Output Shape  |  Parameters   |
| :---         |     :---:      |          ---: |
|flatten_1 (Flatten)  | (None, 784) |  0     |
|dense_1 (Dense)      | (None, 256) | 200960 |   
|dropout_1 (Dropout)  | (None, 256) | 0      |   
|dense_2 (Dense)      | (None, 128) | 32896  |   
|dropout_2 (Dropout)  | (None, 128) | 0      |   
|dense_3 (Dense)      | (None, 100) | 12900  |   
|dropout_3 (Dropout)  | (None, 100) | 0      |   
|dense_4 (Dense)      | (None, 10)  | 1010   |

## Log History

In this repository it is provided in [logs folder](https://github.com/heitorrapela/fashion-mnist-mlp/blob/master/logs/runs_history.txt) the run_history.txt with 30 runs of this architecture where you can check the test with this hyperparameters and model (Only the best one was saved :P )

![Accuracy Curve](https://github.com/heitorrapela/fashion-mnist-mlp/blob/master/logs/accuracy.png)

![Loss Curve](https://github.com/heitorrapela/fashion-mnist-mlp/blob/master/logs/loss.png)

## If you have a question, ask me :)
