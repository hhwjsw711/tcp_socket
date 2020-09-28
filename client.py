# 发送端
import socket
import os
import sys
import struct

ip_port = ("192.168.1.11", 8000)  # 指定要发送的服务器地址和端口


def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 生成socket连接对象
        s.connect(ip_port)  # 连接
    except socket.error as msg:
        print(msg)  # 输出错误信息
        sys.exit(1)
    print("服务器已连接...")
    LEN = 0
    while 1:
        filepath = input("please input the file path:")  # 输入要发送的文件的路径
        if os.path.isfile(filepath):  # 如果文件存在
            # 定义文件名和文件大小
            fhead = struct.pack('128sq', os.path.basename(filepath).encode('utf-8'),
                                os.stat(filepath).st_size)
            s.send(fhead)  # 发送文件名、文件大小等信息
            print('即将发送的文件的路径为：{0}'.format(filepath))
            LENS = os.stat(filepath).st_size  # 获取文件的大小
            fp = open(filepath, 'rb')  # 读取文件
            while 1:
                data = fp.read(1024)
                data_len = len(data)
                LEN += data_len
                if not data:
                    print('{0} 文件发送完毕...'.format(filepath))
                    break
                s.send(data)  # 发送文件
                # print('已发送：', int(LEN / LENS * 100), '%')
            fp.close()  # 关闭
            # 等待服务端传回的数据
            while True:
                fileinfo_size = struct.calcsize('128sq')
                # 接收文件
                buf = s.recv(fileinfo_size)
                if buf:
                    filename, filesize = struct.unpack('128sq', buf)
                    fn = filename.strip('\00'.encode('utf-8'))
                    new_filename = os.path.join('./'.encode('utf-8'), 'new_'.encode('utf-8') + fn)
                    print('文件的新名字是{0}，文件的大小为{1}'.format(new_filename, filesize))

                    recvd_size = 0

                    fp = open(new_filename, 'wb')
                    print('开始接收文件...')

                    while recvd_size < filesize:
                        if filesize - recvd_size > 1024:
                            data = s.recv(1024)
                            recvd_size += len(data)
                        else:
                            data = s.recv(filesize - recvd_size)  # 最后一次接收
                            recvd_size += len(data)
                        # print('已接收：', int(recvd_size / filesize * 100), '%')
                        fp.write(data)  # 写入文件
                    fp.close()
                    break
        s.close()
        break


if __name__ == '__main__':
    socket_client()
