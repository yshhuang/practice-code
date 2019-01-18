#include <iostream>

using namespace std;

int zhizhen() {
    int var = 20;   // 实际变量的声明
    int *ip;        // 指针变量的声明

    ip = &var;       // 在指针变量中存储 var 的地址

    cout << "Value of var variable: ";
    cout << var << endl;

    // 输出在指针变量中存储的地址
    cout << "Address stored in ip variable: ";
    cout << ip << endl;

    // 访问指针中地址的值
    cout << "Value of *ip variable: ";
    cout << *ip << endl;

    return 0;
}

int main() {
    printf("hhh\n");
    zhizhen();
    return 0;
}

int max(int num1, int num2) {
    // 局部变量声明
    int result;

    if (num1 > num2)
        result = num1;
    else
        result = num2;

    return result;
}

