/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: date
Description:
The write a console program that continuously takes in a natural number from the console 
and outputs to the console all the numbers below the entered number that are multiples of 3 or 5 
and then outputs the sum of all the multiples.
*/

#include <iostream>
#include <string>
#include <sstream>
using namespace std;

/*
  The checkNum function is used to check if the input is a number.
  The input is a string. And the function will return true (is a number) or false (not a number).
*/
bool checkNum(string str)
{
    int len = str.length();     //get the length of the string
    bool flag = true;       //define a variable of bool type and initializes it to true. True means the string is a number.
    //use a loop to check for each character
    for (int i = 0; i < len; i++)
    {
        if (!isdigit(str[i]))
        {
            flag = false;
            break;
        }
    }
    return flag;    //return the result of checking
}

/*The main function is used to achieve main function of Problem2.
*/
int main()
{
    string sNum = "";   //define a variable of string type and initializes it to null
    
    //Start an endless loop
    /*For each round,
    1. let user input a string
    2. check if the input is a number. if it is, go to step3. if it isnot, start a new round of the loop.
    3. check if the number is 0. if it is, end the loop and output prompts. if it isnot, go to step4.
    4. use the loop to find multiples of 3 and 5 below the number, and output each number and sum of them. then start a new round.
    */
    while (1)
    {
        cout << "Please enter a natural number (0 to quit): ";
        cin >> sNum;
        //check if the string is a number
        if (checkNum(sNum))
        {
            stringstream ss(sNum);  //define a variable of stringstream type and initializes it to sNum
            int iNum;   //define a variable of int type. 
            ss >> iNum; //Convert the string type to int.
            //check if the number is zero
            if (iNum == 0)  // is zero
            {
                cout << "Program terminated.\nHave a nice day!" << endl;
                return 0;
            }
            else
            {
                int tSum = 0;   //define a variable of int type and initializes it to zero
                cout << "The multiples of 3 below " << iNum << " are: ";
                //find the multiples of 3 below the numer
                for (int i = 3; i < iNum; i = i + 3)
                {
                    tSum += i;
                    if (i != 3)
                    {
                        cout << ", " << i;
                    }
                    else
                    {
                        cout << i;
                    }
                }
                cout << "." << endl;

                cout << "The multiples of 5 below " << iNum << " are: ";
                //find the multiples of 5 below the numer
                for (int i = 5; i < iNum; i = i + 5)
                {
                    tSum += i;
                    if (i != 5)
                    {
                        cout << ", " << i;
                    }
                    else
                    {
                        cout << i;
                    }
                }
                cout << "." << endl;
                //output the sum
                cout << "The sum of all multiples is: " << tSum << "." << endl;
            }
        }
    }
}