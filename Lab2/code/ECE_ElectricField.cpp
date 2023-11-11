/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 10/06/2923
Description:
Lab2: ECE_ElectricField class cpp file for defining member functions.
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
	double dx = x - this->x;
	double dy = y - this->y;
	double dz = z - this->z;
	double r = sqrt(dx * dx + dy * dy + dz * dz);
	Ex = K * q * C * dx / (r * r * r);
	Ey = K * q * C * dy / (r * r * r);
	Ez = K * q * C * dz / (r * r * r);
}

void ECE_ElectricField::getElectricField(double& Ex, double& Ey, double &Ez) 
{
	Ex = this->Ex;
	Ey = this->Ey;
	Ez = this->Ez;
}