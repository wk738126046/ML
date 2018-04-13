# 1. I used mxnet for my ML
 
 
 ##1.1 kaggle first param:
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