#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>

bool split(const std::string& s, char delimiter, std::vector<std::string>& results) 
{
    bool bRC = true;
    if (s.empty())
    {
        bRC = false;
    }
    else
    {
        std::string token;
        std::istringstream tokenStream(s);

        while (std::getline(tokenStream, token, delimiter)) 
        {
            results.push_back(token);
        }
    }

    return bRC;
}


int spltTestmain() 
{
    std::string str = "Hello World From C++";
    char delimiter = ' ';

    std::vector<std::string> result;
    split(str, delimiter, result);
    for (const auto& word : result) 
    {
        std::cout << word << std::endl;
    }

    return 0;
}
// ***********************************************************************
// ***********************************************************************
// ***********************************************************************

void CalculateElectricField(int id)
{
    do
    {
        // Check if calculations should be started or if the thread should exit

        // Do its part of the calculation

        // Signal that it is done with its part

    } while (true);
}

int main22()
{
    // Determine the number of threads that can run concurrently
    unsigned int n = std::thread::hardware_concurrency();
    // Prompt the user for the size of the array and make sure it is valid
    // Prompt the user for the separation distances and make sure it is valid
    // Prompt the user for the charge and make sure it is valid

    // I would create the threads (n-1) once and then signal them when a new location has been entered.
    // Each thread should know which points it should be responsible for and calculate the electrc
    // field contributions

    // Remember to calculate how long it takes do the calculation each time a user enters a location.

    // The main thread can call the thread function to do its part.
    // After the function returns in the main, check the status of the other threads
    // Accumulate the results and display to the user, then prompt the user again for location

    return 0;
}

void HowToCreate2DArray() // using std::vector
{
    // Define the dimensions of the matrix
    int rows = 5;
    int cols = 4;

    // Create a 2D vector with the given dimensions
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    // Fill the matrix with some values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = i * j;  // just a sample value
        }
    }
}


