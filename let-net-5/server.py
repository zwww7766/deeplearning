# -*- coding: utf-8 -*-
import threading
import time

import sockets.config as c
from sockets.websocket import Getmsglist
from sockets.websocket import Msgtoclient
from sockets.websocket import msgGet
from sockets.websocket import websocket_server
import main


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
        result = main.tran_once(int(data))
        arr =[]
        for i in range(len(result)):
            arr.append(result[i][0])

        print arr
        Msgtoclient(uuid,k+':'+str(arr.index(max(arr))))
    elif(k == 'dfdsjjfds'):
        Msgtoclient(uuid, k +":"+ c.config['status'])


# threading.Thread(target=websocketInit, name='Thread-Websocket').start()
main.train_and_evaluate()
time.sleep(0.2)
# socketCheck()

    
