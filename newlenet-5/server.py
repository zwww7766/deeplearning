# -*- coding: utf-8 -*-
import threading
import time

import sockets.config as c
from sockets.websocket import Getmsglist
from sockets.websocket import Msgtoclient
from sockets.websocket import msgGet
from sockets.websocket import websocket_server
from train import *

def websocketInit():
    server = websocket_server('0.0.0.0', 5000)
    server.start()
    print 'listen on 5000 '


def socketCheck():
    print '-----socket check------'
    while True:
        if len(Getmsglist()) > 0:
            ExecuteClientMsg(msgGet())
        time.sleep(0.03)


def ExecuteClientMsg(v):
    print('---one  msg--->%s')
    k = v[1].split(':',1)[0]
    data = v[1].split(':',1)[1]
    uuid = v[0]
    if(k == 'd2q32r2weq23'):
        print 'tarn key:',data
        result,label = predict(train_covnet,int(data))
        print '%s-:-%s'%(result,str(label))
        Msgtoclient(uuid,k+':'+str(result))
    elif(k == 'dfdsjjfds'):
        Msgtoclient(uuid, k +":"+ c.config['status'])


train_covnet = ConvNet()
train_net(train_covnet,  1, [0.0001, 0.0001], 60000)
test_net(train_covnet,  10000)
time.sleep(0.2)
threading.Thread(target=websocketInit, name='Thread-Websocket').start()
socketCheck()

    
