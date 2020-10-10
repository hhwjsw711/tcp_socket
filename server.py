# 接收端
import socket
import threading
import os
import struct
import io
import time

ip_port = ("192.168.1.8", 8000)  # 定义监听地址和端口


def socket_service():
    try:
        # 定义socket连接对象
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 解决端口重用问题
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(ip_port)  # 绑定地址
        s.listen(1)  # 等待最大客户数
    except socket.error as msg:
        print(msg)  # 输出错误信息
        exit(1)
    print('监听开始...')

    while 1:
        conn, addr = s.accept()  # 等待连接
        # 多线程开启
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()


def deal_data(conn, addr):
    print('接收的文件来自{0}'.format(addr))

    while True:
        fileinfo_size = struct.calcsize('128sq')
        # 接收文件
        buf = conn.recv(fileinfo_size)
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
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)  # 最后一次接收
                    recvd_size += len(data)
                # print('已接收：', int(recvd_size / filesize * 100), '%')
                fp.write(data)  # 写入文件
            fp.close()
            # 发送文件
            LEN = 0
            while 1:
                process_filename = str(new_filename, encoding='utf-8')
                if os.path.isfile(process_filename):  # 如果文件存在
                    # 定义文件信息，128sq（其中sq是在不同机器上的衡量单位）表示文件命长128byte
                    fileinfo_size = struct.calcsize('128sq')
                    # 定义文件名和文件大小
                    fhead = struct.pack('128sq', os.path.basename(process_filename).encode('utf-8'),
                                        os.stat(process_filename).st_size)
                    conn.send(fhead)  # 发送文件名、文件大小等信息
                    print('即将发送的文件的路径为：{0}'.format(process_filename))
                    LENS = os.stat(process_filename).st_size  # 获取文件的大小
                    fp = open(process_filename, 'rb')  # 读取文件
                    while 1:
                        data = fp.read(1024)
                        data_len = len(data)
                        LEN += data_len
                        if not data:
                            print('{0} 文件发送完毕...'.format(process_filename))
                            break
                        conn.send(data)  # 发送文件
                        # print('已发送：', int(LEN / LENS * 100), '%')
                    fp.close()  # 关闭
                    break
        conn.close()
        break


def service():
    try:
        # 定义socket连接对象
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 解决端口重用问题
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(ip_port)  # 绑定地址
        s.listen(1)  # 等待最大客户数
    except socket.error as msg:
        print(msg)  # 输出错误信息
        exit(1)
    print('监听开始...')

    while 1:
        conn, addr = s.accept()  # 等待连接

        print('接收的图片来自{0}'.format(addr))

        while True:
            fileinfo_size = struct.calcsize('!H2B2IBI6f2HB')

            # 接收数据
            buf = conn.recv(fileinfo_size)
            if buf:
                head, id, camera_id, image_h, image_v, pixel_size, focal_length, \
                camera_x, camera_y, camera_z, camera_a, camera_b, camera_c, \
                second, millisecond, optic_electronic_hybrid_ban = struct.unpack('!H2B2IBI6f2HB', buf)
                print('帧头为{0}，标识字为{1}，相机标识为{2}，图像横向像素个数为{3}，图像竖向像素个数为{4}，像元大小为{5}，'
                      '焦距为{6}，相机X轴为{7}，相机Y轴为{8}，相机Z轴为{9}，相机滚动角为{10}，相机俯仰角为{11}，'
                      '相机偏航角为{12}，拍照时间秒为{13}，拍照时间毫秒为{14}，'
                      '电光融合准禁为{15}'.format(head, id, camera_id, image_h, image_v, pixel_size, focal_length,
                                           camera_x, camera_y, camera_z, camera_a, camera_b, camera_c,
                                           second, millisecond, optic_electronic_hybrid_ban))

                binary_stream = io.BytesIO()
                while True:
                    image_data = conn.recv(8192)
                    if image_data == bytes('EOF', encoding='utf-8'):
                        print('接收图片成功')
                        break
                    binary_stream.write(image_data)  # 写入内存

                mutable_value = binary_stream.getvalue()

                from new_demo5 import demo
                img_stream = demo(mutable_value, 89, 6912, 640, 6912)

                binary_stream.close()

                binary_stream = io.BytesIO(img_stream)
                while True:
                    image_data = binary_stream.read(8192)
                    if not image_data:
                        print('发送图片成功')
                        break
                    conn.send(image_data)
                binary_stream.close()

                time.sleep(1)
                conn.sendall(bytes('EOF', encoding='utf-8'))

            conn.close()
            break


if __name__ == '__main__':
    service()
