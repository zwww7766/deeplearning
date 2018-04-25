# -*- coding:utf8 -*-

import threading
import hashlib
import socket
import base64
import math
import time

global clients
clients = {}
global msglist
msglist = []

# 通知客户端//广播


def notify(message):
    for connection in clients.values():
        connection.sendall('%c%c%s' % (0x81, len(message), message))


# 通知单体客户端
def Msgtoclient(uuid,message):
    print '-----------send------------msglen:>:',message
    clients[uuid].send('%c%c%s' % (0x81, len(message), message))
        # 返回客户端组

def msgGet():
    global msglist
    single = msglist.pop(0)
    return single

def Getclients():
    global clients
    return clients
# 返回消息组


def Getmsglist():
    global msglist
    return msglist
# 清空消息组


def Cleanmsglist():
    global msglist
    msglist = []
# 客户端处理线程


class websocket_thread(threading.Thread):
    def __init__(self, connection, username):
        super(websocket_thread, self).__init__()
        self.connection = connection
        self.username = username

    def run(self):
        print 'new websocket client joined!'
        data = self.connection.recv(1024)
        headers = self.parse_headers(data)
        token = self.generate_token(headers['Sec-WebSocket-Key'])
        self.connection.send('\
HTTP/1.1 101 WebSocket Protocol Hybi-10\r\n\
Upgrade: WebSocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Accept: %s\r\n\r\n' % token)
        while True:
            try:
                data = self.connection.recv(1024)
            except socket.error, e:
                print "unexpected error: ", e
                clients.pop(self.username)
                break
            data = self.parse_data(data)
            # ----------------------------------------------------------简单的立即通知，不做全异步
            print '------->------>:'+data
            data = data.split(":", 1)
            clients[data[0]] = self.connection
            msglist.append([data[0],data[1]])

            # d2q32r2weq23:data:image / octet - stream;
            # base64, iVBORw0KGgoAAAANSUhEUgAAAMgAAAEsCAYAAACG + vy + AAAAAXNSR0IArs4c6QAADiZJREFUeAHt3cvLLEcZx / FjNILk4E7 / APGPcOHioK684t8hqBGUgBCNQnAl7sSN / 4
            # VLNSsvG3EnJDtBEDVCMMdLvNYvxz6Z6enueZ7qute3YXjf6anurvo89XRV98w774MHLAgggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCBQUOCH4Vj / CY // bjy0 / rnwYEFgSoG9xFgnC0lyonu868S2bFpH4K1w2Gcdh1bCPOMoT9ELARLkAqODXzVqxMQsZpsOOPJXkTNLfuMUR / ha2IlGAjp6Ck32MZTAkhzrawvP86FAaAwCi0CK5FAisSAwnMBXQ4s8o8ReWV23sCAwlECq5FDScJt3qK5BY1IlB28U0peGE0iRHD8YToUGIRAE3gyPvesIy3q9gciCwJAC / w6tsiTBXhnd7WJBYEgB62eqSI4hw0 + jjgT2Or11PSPHkS6vdS1wduR4oevWU3kEDgTOXnPobhcLAkMK / DO0yjqF2ipHcgzZLWiUBN4Ij61Ob11H

    def parse_data(self, msg):
        v = ord(msg[1]) & 0x7f
        if v == 0x7e:
            p = 4
        elif v == 0x7f:
            p = 10
        else:
            p = 2
        mask = msg[p:p + 4]
        data = msg[p + 4:]
        return ''.join([chr(ord(v) ^ ord(mask[k % 4])) for k, v in enumerate(data)])

    def parse_headers(self, msg):
        headers = {}
        print 'header msg-->'
        print msg
        header, data = msg.split('\r\n\r\n', 1)
        for line in header.split('\r\n')[1:]:
            key, value = line.split(': ', 1)
            headers[key] = value
        headers['data'] = data
        return headers
    # 生成token
    def generate_token(self, msg):
        key = msg + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        ser_key = hashlib.sha1(key).digest()
        return base64.b64encode(ser_key)


# 服务端
class websocket_server(threading.Thread):
    def __init__(self, host, port):
        super(websocket_server, self).__init__()
        self.port = port
        self.host = host
    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(5)
        print 'websocket success '
        while True:
            connection, address = sock.accept()
            try:
                username = "ID" + str(address[1])
                thread = websocket_thread(connection, username)
                thread.start()
                print 'usr',username
            except socket.timeout:
                print 'websocket connection timeout!'
