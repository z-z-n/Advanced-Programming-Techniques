/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 09/23/2923
Description:
Lab1: Header file for the ECE_ElectricField class to define the class structure,
member variables and member functions. This class inherits from the ECE_PointCharge class.
*/
#ifndef ECE_ElectricField_h
#define ECE_ElectricField_h

#include "ECE_PointCharge.h"
#include <math.h>

using namespace std;
const double K = 9 * 1e9;
const double C = 1e-6;

class ECE_ElectricField: public ECE_PointCharge
{
public:
	ECE_ElectricField(double x, double y, double z, double q);
	void computeFieldAt(double x, double y, double z);
	void getElectricField(double& Ex, double& Ey, double &Ez);
private:
	double Ex; // Electric field in the x-direction.
	double Ey; // Electric field in the y-direction.
	double Ez; // Electric field in the z-direction.
};

#endif 