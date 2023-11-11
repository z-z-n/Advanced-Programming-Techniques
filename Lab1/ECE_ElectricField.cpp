/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 09/23/2923
Description:
Lab1: ECE_ElectricField class cpp file for defining member functions.
*/
#include "ECE_ElectricField.h"


ECE_ElectricField::ECE_ElectricField(double x, double y, double z, double q) :ECE_PointCharge(x,y,z,q)
{
	Ex = 0.0;
	Ey = 0.0;
	Ez = 0.0;
}

void ECE_ElectricField::computeFieldAt(double x, double y, double z) 
{
	double r = sqrt(pow(this->x - x, 2) + pow(this->y - y, 2) + pow(this->z - z, 2));
	Ex = K * q * C * (x - this->x) / pow(r, 3);
	Ey = K * q * C * (y - this->y) / pow(r, 3);
	Ez = K * q * C * (z - this->z) / pow(r, 3);
}

void ECE_ElectricField::getElectricField(double& Ex, double& Ey, double &Ez) 
{
	Ex = this->Ex;
	Ey = this->Ey;
	Ez = this->Ez;
}