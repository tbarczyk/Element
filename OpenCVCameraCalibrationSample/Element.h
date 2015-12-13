#pragma once
#ifndef FCABLIB
#include "file_calibration.h"
#endif

class Element
{
public:
	
	Element();
	void ComputeElement(calibrationResult);
	void GenerateElements(int, int);
	~Element();
};

