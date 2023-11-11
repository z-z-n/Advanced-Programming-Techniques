/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 10/06/2923
Description:
Lab2: ECE_PointCharge class cpp file for defining member functions.
*/
#include "ECE_PointCharge.h"

ECE_PointCharge::ECE_PointCharge(double x1, double y1, double z1, double q1) :x(x1), y(y1), z(z1), q(q1) {}
void ECE_PointCharge::setLocation(double x1, double y1, double z1)
{
	x = x1;
	y = y1;
	z = z1;
}
void ECE_PointCharge::setCharge(double q1)
{
	q = q1;
}