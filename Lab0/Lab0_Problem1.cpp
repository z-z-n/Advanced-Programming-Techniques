/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 09/02/2023
Description:
Write a C++ program using the insertion stream operator and escape sequences
*/

#include <iostream>

/*The main function to output information.*/
int main()
{
    std::cout << "My name is: Zhining Zhang"<< std::endl;
    std::cout << "This (\") is a double quote." << std::endl;
    std::cout << "This (\') is a single quote." << std::endl;
    std::cout << "This (\\) is a backslash." << std::endl;
    std::cout << "This (/) is a forward slash." << std::endl;
    return 0;
}
