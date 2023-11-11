/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 09/23/2923
Description:
Lab1: Header file for the ECE_PointCharge class to define the class structure,
member variables and member functions. 
*/
#ifndef ECE_PointCharge_h
#define ECE_PointCharge_h
using namespace std;

class ECE_PointCharge
{
public:
	ECE_PointCharge(double x, double y, double z, double q);
	void setLocation(double x, double y, double z);
	void setCharge(double q);

protected:
	double x; // x-coordinate.
	double y; // y-coordinate.
	double z; // z-coordinate.
	double q; // charge of the point.
};

#endif 
