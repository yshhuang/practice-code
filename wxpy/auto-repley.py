import wxpy 

bot = wxpy.Bot()

# 获取好友
tong = bot.friends().search('桐')[0]

#  注册获得个人的图灵机器人key 填入
tuling = wxpy.Tuling(api_key='53bb9a8dfa860975')


# 使用图灵机器人自动与指定好友聊天
@bot.register(tong)
def reply_my_friend(msg):
    print(msg)
    tuling.do_reply(msg)


wxpy.embed()