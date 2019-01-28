# 导入模块
import wxpy
# 初始化机器人，扫码登陆
bot = wxpy.Bot()
my_friend = bot.friends().search('李艾桐', sex=wxpy.MALE)[0]
print(my_friend)
