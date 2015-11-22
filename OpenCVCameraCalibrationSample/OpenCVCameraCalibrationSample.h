// OpenCVCameraCalibrationSample.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// COpenCVCameraCalibrationSampleApp:
// See OpenCVCameraCalibrationSample.cpp for the implementation of this class
//

class COpenCVCameraCalibrationSampleApp : public CWinApp
{
public:
	COpenCVCameraCalibrationSampleApp();

// Overrides
	public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern COpenCVCameraCalibrationSampleApp theApp;