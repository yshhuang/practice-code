# encoding:utf-8
"""
@Time    : 2020-05-06 10:04
@Author  : yshhuang@foxmail.com
@File    : secret.py
@Software: PyCharm
"""
# coding=utf-8
import hmac, base64, struct, hashlib, time


# 参数secretKey是开通google身份验证时的密钥
def calGoogleCode(secretKey):
    input = int(time.time()) // 30
    key = base64.b32decode(secretKey)
    msg = struct.pack(">Q", input)
    googleCode = hmac.new(key, msg, hashlib.sha1).digest()
    o = ord(googleCode[19]) & 15
    googleCode = str((struct.unpack(">I", googleCode[o:o + 4])[0] & 0x7fffffff) % 1000000)
    if len(googleCode) == 5:  # 如果验证码的第一位是0，则不会显示。此处判断若是5位码，则在第一位补上0
        googleCode = '0' + googleCode
    return googleCode


if __name__ == '__main__':
    calGoogleCode('930401')
