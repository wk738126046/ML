# 1. I used mxnet for my ML
 
 
## 1.1 kaggle first param( house price )

    k = 5
    epochs = 100 
    verbose_epoch = 95
    learning_rate = 0.3
    weight_decay = 3.0

##1.2 net   

     net.add(gluon.nn.Dense(128))
     net.add(gluon.nn.BatchNorm(),
             gluon.nn.Activation('relu'))
     net.add(gluon.nn.Dense(1))

##2.1 secend param

    k = 5
    epochs = 50
    verbose_epoch = 45
    learning_rate = 0.03
    weight_decay = 170

##2.2 net

    net.add(gluon.nn.Dense(1024, activation='relu'))
    net.add(gluon.nn.Dropout(0.5))
    net.add(gluon.nn.Dense(1))
    
##3.1 CIFAR10 param

###3.1.1 firts param( epoch 160 and use lr_decay )
   ```
    num_epochs = 300
    learning_rate = 0.1
    weight_decay = 0.0005
    lr_period = 40
    lr_decay = 0.5
``` 

    
