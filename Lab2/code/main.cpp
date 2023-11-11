/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 10/06/2923
Description:
Lab2: Use OpenMp to calculate the electric field at a certain location.
*/

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <regex>
#include <chrono>
#include <iomanip>
#include "omp.h"
#include "math.h"

#include "ECE_ElectricField.h"
#include "ECE_PointCharge.h"

using namespace std;
using namespace chrono;

// gobal variables
double sumEx = 0, sumEy = 0, sumEz = 0;

// input string, kinds of checking
bool checkNum(string str, int type)
{
    string nature{ "[1-9][0-9]*" };
    string pDigital{ "[0-9]+.[0-9]+" };
    string digital{ "-[0-9]+.[0-9]+|[0-9]+.[0-9]+" };
    string zero{ "[0]+.[0]+" };
    regex reNt(nature);
    regex rePDg(pDigital);
    regex reDg(digital);
    regex reZo(zero);

    if (type == 0)
    {
        //Is it a natural number? eg:N M
        return regex_match(str, reNt);
    }
    else if (type == 1)
    {
        if (regex_match(str, reZo))
        {
            return false;
        }
        //Is it a positive decimal ? distance x y
        return regex_match(str, rePDg);
    }
    else
    {
        //Is it a natural number or decimal ? xyz location
        return (regex_match(str, reDg) || regex_match(str, reNt));
    }
}


bool split(const string& s, char delimiter, vector<string>& results, int len, int type)
{
    bool bRC = true;
    if (s.empty())
    {
        bRC = false;
    }
    else
    {
        string token;
        istringstream tokenStream(s);

        while (getline(tokenStream, token, delimiter))
        {
            results.push_back(token);
            if (checkNum(token, type) == false)
            {
                bRC = false;
                break;
            }
        }

        if (results.size() != len)
        {
            bRC = false;
        }
        if (!bRC)
        {
            cout << "Invalid Inputs!" << endl;
        }
    }
    return bRC;
}

//Convert double to scientific notation
pair<double, int> Formatted(double num)
{
    pair<double, int> results;
    int y = round(floor(log10(abs(num))));
    double tNum = num * pow(10, -y);
    tNum = round(tNum * 10000) / 10000;
    results = make_pair(tNum, y);
    return results;
}

int main()
{
    string tNums;
    char deM = ' ';
    vector<string> splitResults;
    // Prompt the user for the number of threads that can run concurrently
    do
    {
        splitResults.clear();
        cout << "Please enter the number of concurrent threads to use: ";
        getline(cin, tNums);

    } while (!split(tNums, deM, splitResults, 1, 0));
    int n = stoi(splitResults[0]);

    // Prompt the user for the size of the array and make sure it is valid
    do
    {
        splitResults.clear();
        cout << "Please enter the number of rows and columns in the N x M array: ";
        getline(cin, tNums);

    } while (!split(tNums, deM, splitResults, 2, 0));
    int N = stoi(splitResults[0]);
    int M = stoi(splitResults[1]);

    // Prompt the user for the separation distances and make sure it is valid
    do
    {
        splitResults.clear();
        cout << "Please enter the x and y separation distances in meters: ";
        getline(cin, tNums);

    } while (!split(tNums, deM, splitResults, 2, 1));
    double x0 = stod(splitResults[0]);
    double y0 = stod(splitResults[1]);

    // Prompt the user for the charge and make sure it is valid
    do
    {
        splitResults.clear();
        cout << "Please enter the common charge on the points in micro C: ";
        getline(cin, tNums);

    } while (!split(tNums, deM, splitResults, 1, 1));
    double q = stod(splitResults[0]);

    // I would create the threads (n-1) once and then signal them when a new location has been entered.
    if (n > M * N)
    {
        n = M * N;
    }
    vector<thread> calThreads(n - 1);
    bool flag = true;
    // begin the loop of changing the location
    do
    {
        // Prompt the user for the location and make sure it is valid
        do
        {
            splitResults.clear();
            cout << "Please enter the location in space to determine the electric field (x y z) in meters: ";
            getline(cin, tNums);

        } while (!split(tNums, deM, splitResults, 3, 2));
        double x = stod(splitResults[0]);
        double y = stod(splitResults[1]);
        double z = stod(splitResults[2]);

        // Check the input of x,y,z is valid
        bool xyzValid = true;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                double xCur = -(((double)N - 1) * x0) / 2 + x0 * i;
                double yCur = -(((double)M - 1) * y0) / 2 + y0 * j;
                if (z == 0.0 && x == xCur && y == yCur)
                {
                    xyzValid = false;
                    cout << xCur << " " << yCur << endl;
                }
            }
        }
        if (!xyzValid)
        {
            cout << "Invalid Inputs!" << endl;
            continue;
        }

        // create class ECE_ElectricField objects
        vector<ECE_ElectricField> eFields;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                ECE_ElectricField tField(-(((double)N - 1) * x0) / 2 + x0 * i, -(((double)M - 1) * y0) / 2 + y0 * j, 0, q);
                eFields.push_back(tField);
            }
        }

        //set the number of opemMp threads
        omp_set_num_threads(n);
        high_resolution_clock::time_point startTime, endTime;
#pragma omp parallel
        {
            // Local private variables
            double tEx, tEy, tEz;
#pragma omp master
            {
                startTime = std::chrono::high_resolution_clock::now();
            }
#pragma omp for reduction(+:sumEx, sumEy, sumEz) schedule(static)
            for (int i = 0; i < eFields.size(); i++)
            {
                eFields[i].computeFieldAt(x, y, z);
                eFields[i].getElectricField(tEx, tEy, tEz);
                sumEx += tEx;
                sumEy += tEy;
                sumEz += tEz;
            }
#pragma omp master
            {
                endTime = std::chrono::high_resolution_clock::now();
            }
        }
        auto duration = duration_cast<microseconds>(endTime - startTime);

        // After the function returns in the main, check the status of the other threads
        double E = sqrt(pow(sumEx, 2) + pow(sumEy, 2) + pow(sumEz, 2));
        cout << "The electric field at (" << x << ", " << y << ", " << z << ") in V / m is" << endl;
        cout << "Ex = " << fixed << setprecision(4) << Formatted(sumEx).first << " * 10^" << Formatted(sumEx).second << endl;
        cout << "Ey = " << fixed << setprecision(4) << Formatted(sumEy).first << " * 10^" << Formatted(sumEy).second << endl;
        cout << "Ez = " << fixed << setprecision(4) << Formatted(sumEz).first << " * 10^" << Formatted(sumEz).second << endl;
        cout << "|E| = " << fixed << setprecision(4) << Formatted(E).first << " * 10^" << Formatted(E).second << endl;

        cout << "The calculation took " << duration.count() << " microsec!" << endl;

        // Accumulate the results and display to the user, then prompt the user again for location
        do
        {
            string yOrN;
            cout << "Do you want to enter a new location (Y/N)? ";
            getline(cin, yOrN);
            if (yOrN == "N")
            {
                cout << "Bye!" << endl;
                flag = false;
                break;
            }
            else if (yOrN == "Y")
            {
                break;
            }
            else
            {
                cout << "Invalid Inputs!" << endl;
            }
        } while (true);

    } while (flag);

    return 0;
}