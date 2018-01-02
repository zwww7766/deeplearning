# -*- coding:utf8 -*-
import threading
import time
from websocket import websocket_server
from websocket import Getmsglist
from websocket import msgGet
from websocket import Msgtoclient
import network.mainmain as adm
import config as config

def websocketInit():
    server = websocket_server('0.0.0.0', 5000)
    server.start()
    print 'listen on 5000 '


def socketCheck():
    print 'socket check'
    while True:
        if len(Getmsglist()) > 0:
            ExecuteClientMsg(msgGet())
        time.sleep(0.03)


def ExecuteClientMsg(v):
    print('---one  msg--->%s', v[1])
    k, j = v[1].split(':',1)[0]
    uuid = j[0]
    data = j[1]
    if(k == 'd2q32r2weq23'):
        data = data.split(',')
        print j
        print len(j)
        Msgtoclient(uuid,k+adm.predict_sample(j))
    elif(k == 'dfdsjjfds'):
        Msgtoclient(uuid, k+config['status'])

threading.Thread(target=websocketInit, name='Thread-Websocket').start()
time.sleep(0.2)
socketCheck()
adm.train_and_evaluate()
    
