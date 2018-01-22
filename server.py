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
    k = v[1].split(':',1)[0]
    data = v[1].split(':',1)[1]
    uuid = v[0]
    if(k == 'd2q32r2weq23'):
        data = data.split(',')
        Msgtoclient(uuid,k+adm.predict_sample(data))
    elif(k == 'dfdsjjfds'):
        Msgtoclient(uuid, k +":"+ c.config['status'])


threading.Thread(target=websocketInit, name='Thread-Websocket').start()
adm = admin()
adm.train_and_evaluate()
time.sleep(0.2)
socketCheck()

    
