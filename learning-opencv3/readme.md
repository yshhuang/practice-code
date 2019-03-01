learning opencv 3 的示例代码
 ## 编译命令
  g++ $(pkg-config --cflags --libs opencv4) -std=c++14 display-video.cpp -o out/display-video.out
  
 ## 执行程序
 $(pwd)/out/display-video.out $(pwd)/resources/music.mp4