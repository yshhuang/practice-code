#!/bin/bash
#此脚本用于在shell环境下操作MySQL数据库
export MYSQL_PWD=root1234
MYSQL="mysql -hlocalhost -uroot  --default-character-set=utf8 -A -N"
#这里面有两个参数，-A、-N，-A的含义是不去预读全部数据表信息，这样可以解决在数据表很多的时候卡死的问题
#-N，很简单，Don't write column names in results，获取的数据信息省去列名称

#0.清空数据
sql="truncate kuaikan.topic"
$MYSQL -e "$sql"

# 1.添加数据
OLF_IFS=$IFS
IFS=" "
cat topic.csv |while read id title url
do 
    # echo $title
    sql="insert into kuaikan.topic(id,title,topic_url) value(${id},'${title}','${url}');"
    # echo $sql
    $MYSQL -e "$sql"
done

#2.删除数据
sql="delete from kuaikan.topic where id=300;"
$MYSQL -e "$sql"

#3.修改数据
sql="update kuaikan.topic  set title='400' where id=400;"
$MYSQL -e "$sql"

#4.查询数据
sql="select * from kuaikan.topic where id=400"
result="$($MYSQL -e "$sql")"
echo $result 
echo $MYSQL_PWD