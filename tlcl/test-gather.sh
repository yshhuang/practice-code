#!/bin/bash
#此脚本用于在shell环境下操作MySQL数据库
export MYSQL_PWD=root1234
MYSQL="mysql -hlocalhost -uroot  --default-character-set=utf8 -A -N"
$MYSQL -e "use xm_search;"

function isSameGather(){
        sql1="select topic_gather_id from xm_search.vs_topic where topic_url='$1';"
        sql2="select topic_gather_id from xm_search.vs_topic where topic_url='$2';"

        result1="$($MYSQL -e "$sql1")"
        result2="$($MYSQL -e "$sql2")"
        echo $result1
        if [ $result1 = $result2 ]&&[ -n "$result1" ]
        then
            return 1
        else
            return 0
        fi
}

# OLF_IFS=$IFS
# IFS=" "
all=0
correct=0
while read url1 url2
do
        isSameGather $url1 $url2
        if [ $? -eq 1 ]
        then
            correct=$(($correct+1))
        fi
        all=$(($all+1))  
done < gather.txt 

echo "在同一集合中的topic比例为:"$correct/$all=`awk 'BEGIN{printf "%.1f%%\n",('$correct'/'$all')*100}'`
# echo $correct/$all


