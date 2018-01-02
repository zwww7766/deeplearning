# -*- coding:utf8 -*-
import threading
import time
from sockets.websocket import websocket_server
from sockets.websocket import Getmsglist
from sockets.websocket import msgGet
from sockets.websocket import Msgtoclient
from network.mainmain import admin
import sockets.config as c

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
        Msgtoclient(uuid, k + c.config.config['status'])

adm = admin()
adm.train_and_evaluate()
threading.Thread(target=websocketInit, name='Thread-Websocket').start()
time.sleep(0.2)
socketCheck()

    
