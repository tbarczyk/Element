// OpenCVCameraCalibrationSampleDlg.cpp : implementation file
//

#include "stdafx.h"
#include "OpenCVCameraCalibrationSample.h"
#include "OpenCVCameraCalibrationSampleDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// The following code links to the OpenCV libraries.
// It is important to install the correct version of OpenCV before compiling this project
#ifdef _DEBUG
#	pragma comment (lib, "opencv_highgui2411d.lib")
#	pragma comment (lib, "opencv_core2411d.lib")
#	pragma comment (lib, "opencv_imgproc2411d.lib")
#	pragma comment (lib, "opencv_calib3d2411d.lib")
#else
#	pragma comment (lib, "opencv_highgui245.lib")
#	pragma comment (lib, "opencv_core245.lib")
#	pragma comment (lib, "opencv_imgproc245.lib")
#	pragma comment (lib, "opencv_calib3d245.lib")
#endif

// Select if this sample demonstrates a Warp or a edge-finder
//#define WARPSAMPLE 1

// COpenCVCameraCalibrationSampleDlg dialog

//--------------------------------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------------------------------
COpenCVCameraCalibrationSampleDlg::COpenCVCameraCalibrationSampleDlg(CWnd* pParent /*=NULL*/)
: CDialog(COpenCVCameraCalibrationSampleDlg::IDD, pParent)
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

    m_hFactory = NULL;
    m_hCam = NULL;
    m_hThread = NULL;
    m_BoardSize.width = 9;
    m_BoardSize.height = 5;
    m_SquareSize = 1.f;
    m_AspectRatio = 1.f;
    m_Flags = 0;
    m_ImageCount = 10;

    m_pImg = NULL;

    m_CameraMatrix = cvMat( 3, 3, CV_64F, m_Camera);
    m_DistCoeffsMatrix = cvMat( 1, 5, CV_64F, m_DistCoeffs);
    m_pExtrParamsMatrix = 0;
    m_pReprojErrsMatrix = 0;
    m_AvgReprojRrr = 0;

    m_bCalibrated = false;
    m_bCalibratedChanged = false;
    m_CaptureDelay = 0;

    m_pUndistortMapX = 0;
    m_pUndistortMapY = 0;
    m_pImagePointsBuf = 0;
    m_pImagePointsSeq = 0;
    m_pStorage = 0;
    m_bSaveSettings = false;
    m_bSettingsSaved = false;
    m_bSaveImages = false;
    m_bImagesSaved = false;
    m_bCameraOpen = false;
    m_bAcquisitionRunning = false;

    InitializeCriticalSection(&m_CriticalSection);
}

void COpenCVCameraCalibrationSampleDlg::DoDataExchange(CDataExchange* pDX)
{
    CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(COpenCVCameraCalibrationSampleDlg, CDialog)      
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    //}}AFX_MSG_MAP
    ON_WM_DESTROY()
    ON_BN_CLICKED(IDC_START, &COpenCVCameraCalibrationSampleDlg::OnBnClickedStart)
    ON_BN_CLICKED(IDC_STOP, &COpenCVCameraCalibrationSampleDlg::OnBnClickedStop)
    ON_WM_TIMER()
    ON_WM_HSCROLL()
    ON_BN_CLICKED(IDC_CALIBRATE_BUTTON, &COpenCVCameraCalibrationSampleDlg::OnBnClickedCalibrateButton)
    ON_BN_CLICKED(IDC_SAVE_IMAGE_BUTTON, &COpenCVCameraCalibrationSampleDlg::OnBnClickedSaveImageButton)
    ON_BN_CLICKED(IDC_SAVE_SETTINGS_BUTTON, &COpenCVCameraCalibrationSampleDlg::OnBnClickedSaveSettingsButton)
    ON_NOTIFY(UDN_DELTAPOS, IDC_CHESS_ROWS_SPIN, &COpenCVCameraCalibrationSampleDlg::OnDeltaposChessRowsSpin)
    ON_NOTIFY(UDN_DELTAPOS, IDC_CHESS_COLS_SPIN, &COpenCVCameraCalibrationSampleDlg::OnDeltaposChessColsSpin)
    ON_NOTIFY(UDN_DELTAPOS, IDC_IMAGE_COUNT_SPIN, &COpenCVCameraCalibrationSampleDlg::OnDeltaposImageCountSpin)
END_MESSAGE_MAP()


// COpenCVCameraCalibrationSampleDlg message handlers

//--------------------------------------------------------------------------------------------------
// OnInitDialog
//--------------------------------------------------------------------------------------------------
BOOL COpenCVCameraCalibrationSampleDlg::OnInitDialog()
{
    CDialog::OnInitDialog();

    // Set the icon for this dialog.  The framework does this automatically
    //  when the application's main window is not a dialog
    SetIcon(m_hIcon, TRUE);			// Set big icon
    SetIcon(m_hIcon, FALSE);		// Set small icon

    BOOL retval;

    // Open factory & camera
    retval = OpenFactoryAndCamera();  
    if (retval)
    {
        GetDlgItem(IDC_CAMERAID)->SetWindowText(CString((char*)m_sCameraId));    // Display camera ID
        InitializeControls();   // Initialize Controls
        m_bCameraOpen = true;
    }
    else
    {
        GetDlgItem(IDC_CAMERAID)->SetWindowText(CString("error"));
    }
    EnableControls();   // Enable Controls

    return TRUE;  // return TRUE  unless you set the focus to a control
}
//--------------------------------------------------------------------------------------------------
// OnDestroy
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::OnDestroy()
{
    CDialog::OnDestroy();

    // Stop acquisition
    OnBnClickedStop();

    // Close factory & camera
    CloseFactoryAndCamera();
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void COpenCVCameraCalibrationSampleDlg::OnPaint()
{
    if (IsIconic())
    {
        CPaintDC dc(this); // device context for painting

        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon);
    }
    else
    {
        CDialog::OnPaint();
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR COpenCVCameraCalibrationSampleDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}

void COpenCVCameraCalibrationSampleDlg::OnTimer(UINT_PTR nIDEvent)
{
    // Update the GUI with latest processing time value
    if (nIDEvent == 1)
    {
        // Update UI when calibrated flag changed
        if (m_bCalibratedChanged)
        {
            m_bCalibratedChanged = false;
            EnableControls();
        }
        if (m_bImagesSaved)
        {
            m_bImagesSaved = false;
            AfxMessageBox(_T("Images saved to disk!"), MB_OK|MB_ICONINFORMATION);
        }
        if (m_bSettingsSaved)
        {
            m_bSettingsSaved = false;
            AfxMessageBox(_T("Settings saved to disk!"), MB_OK|MB_ICONINFORMATION);
        }
    }

    CDialog::OnTimer(nIDEvent);
}

//--------------------------------------------------------------------------------------------------
// OpenFactoryAndCamera
//--------------------------------------------------------------------------------------------------
BOOL COpenCVCameraCalibrationSampleDlg::OpenFactoryAndCamera()
{ 
    J_STATUS_TYPE   retval;  
	uint32_t        iSize;
    uint32_t        iNumDev;
    bool8_t         bHasChange;

    // Open factory
    retval = J_Factory_Open((int8_t*)"" , &m_hFactory);
    if (retval != J_ST_SUCCESS)
    {
        AfxMessageBox(CString("Could not open factory!"));
        return FALSE;
    }
	AfxMessageBox(CString("Opening factory succeeded\n"));
    
    // Update camera list
    retval = J_Factory_UpdateCameraList(m_hFactory, &bHasChange);
    if (retval != J_ST_SUCCESS)
    {
        AfxMessageBox(CString("Could not update camera list!"), MB_OK | MB_ICONEXCLAMATION);
        return FALSE;
    }
    TRACE("Updating camera list succeeded\n");

    // Get the number of Cameras
    retval = J_Factory_GetNumOfCameras(m_hFactory, &iNumDev);
    if (retval != J_ST_SUCCESS)
    {
        AfxMessageBox(CString("Could not get the number of cameras!"), MB_OK | MB_ICONEXCLAMATION);
        return FALSE;
    }
    if (iNumDev == 0)
    {
        AfxMessageBox(CString("There is no camera!"), MB_OK | MB_ICONEXCLAMATION);
        return FALSE;
    }
    TRACE("%d cameras were found\n", iNumDev);

    // Get camera ID
    iSize = (uint32_t)sizeof(m_sCameraId);
    retval = J_Factory_GetCameraIDByIndex(m_hFactory, 0, m_sCameraId, &iSize);
    if (retval != J_ST_SUCCESS)
    {
        AfxMessageBox(CString("Could not get the camera ID!"), MB_OK | MB_ICONEXCLAMATION);
        return FALSE;
    }
    TRACE("Camera ID: %s\n", m_sCameraId);

    // Open camera
    retval = J_Camera_Open(m_hFactory, m_sCameraId, &m_hCam);
    if (retval != J_ST_SUCCESS)
    {
        AfxMessageBox(CString("Could not open the camera!"), MB_OK | MB_ICONEXCLAMATION);
        return FALSE;
    }
    TRACE("Opening camera succeeded\n");

    return TRUE;
}
//--------------------------------------------------------------------------------------------------
// CloseFactoryAndCamera
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::CloseFactoryAndCamera()
{
    if (m_hCam)
    {
        // Close camera
        J_Camera_Close(m_hCam);
        m_hCam = NULL;
        TRACE("Closed camera\n");
    }

    if (m_hFactory)
    {
        // Close factory
        J_Factory_Close(m_hFactory);
        m_hFactory = NULL;
        TRACE("Closed factory\n");
    }
}
//--------------------------------------------------------------------------------------------------
// OnBnClickedStart
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::OnBnClickedStart()
{
    J_STATUS_TYPE   retval;
    int64_t int64Val;
    int64_t pixelFormat;

    SIZE	ViewSize;
    POINT	TopLeft;

    // Get Width from the camera
    retval = J_Camera_GetValueInt64(m_hCam, NODE_NAME_WIDTH, &int64Val);
    ViewSize.cx = (LONG)int64Val;     // Set window size cx

    // Get Height from the camera
    retval = J_Camera_GetValueInt64(m_hCam, NODE_NAME_HEIGHT, &int64Val);
    ViewSize.cy = (LONG)int64Val;     // Set window size cy

    // Get pixelformat from the camera
    retval = J_Camera_GetValueInt64(m_hCam, NODE_NAME_PIXELFORMAT, &int64Val);
    pixelFormat = int64Val;

    // Calculate number of bits (not bytes) per pixel using macro
    int bpp = J_BitsPerPixel(pixelFormat);

    // Set window position
    TopLeft.x = 100;
    TopLeft.y = 50;

    // Open stream
    retval = J_Image_OpenStream(m_hCam, 0, reinterpret_cast<J_IMG_CALLBACK_OBJECT>(this), reinterpret_cast<J_IMG_CALLBACK_FUNCTION>(&COpenCVCameraCalibrationSampleDlg::StreamCBFunc), &m_hThread, (ViewSize.cx*ViewSize.cy*bpp)/8);
    if (retval != J_ST_SUCCESS) {
        AfxMessageBox(CString("Could not open stream!"), MB_OK | MB_ICONEXCLAMATION);
        return;
    }
    TRACE("Opening stream succeeded\n");

    // Create two OpenCV named Windows used for displaying "Before" and "After" images
    cvNamedWindow("Source",0);
    cvMoveWindow("Source", 0,0);
    cvResizeWindow("Source", 512,384); 
    cvNamedWindow("Output",0);
    cvMoveWindow("Output", 0,400);
    cvResizeWindow("Output", 512,384); 

    int elem_size = m_BoardSize.width*m_BoardSize.height*sizeof(CvPoint2D32f);
    if (m_pStorage == 0)
    {
        m_pStorage = cvCreateMemStorage(MAX( elem_size*4, 1 << 16 ));
        m_pImagePointsBuf = (CvPoint2D32f*)cvAlloc( elem_size );
        m_pImagePointsSeq = cvCreateSeq( 0, sizeof(CvSeq), elem_size, m_pStorage);
    }

    // Start Acquision
    retval = J_Camera_ExecuteCommand(m_hCam, NODE_NAME_ACQSTART);

    m_bAcquisitionRunning = true;
    SetTimer(1, 500, NULL);
    EnableControls();
}
//--------------------------------------------------------------------------------------------------
// StreamCBFunc
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::StreamCBFunc(J_tIMAGE_INFO * pAqImageInfo)
{
    CString valueString;

    // Skip images if they start to queue up in order to avoid lag caused by the queue size.
    if (pAqImageInfo->iAwaitDelivery > 2)
        return;

    // We only want to create the OpenCV Image object once and we want to get the correct size from the Acquisition Info structure
    if (m_pImg == NULL)
    {
        // Create the Image:
        // We assume this is a 8-bit monochrome image in this sample
        m_pImg = cvCreateImage(cvSize(pAqImageInfo->iSizeX,pAqImageInfo->iSizeY), IPL_DEPTH_8U, 1);

        // Create Undistort maps
        m_pUndistortMapX = cvCreateImage(cvSize(pAqImageInfo->iSizeX,pAqImageInfo->iSizeY), IPL_DEPTH_32F, 1);
        m_pUndistortMapY = cvCreateImage(cvSize(pAqImageInfo->iSizeX,pAqImageInfo->iSizeY), IPL_DEPTH_32F, 1);

        m_ImgSize = cvGetSize(m_pImg);
    }

    // Copy the data from the Acquisition engine image buffer into the OpenCV Image obejct
    memcpy(m_pImg->imageData,pAqImageInfo->pImageBuffer,m_pImg->imageSize);

    int count = 0;
    int found = 0;
    int index = 0;

    if (!m_bCalibrated)
    {
        // Look in all three images for the chessboard pattern
        found = cvFindChessboardCorners(m_pImg, 
                                        m_BoardSize, 
                                        m_pImagePointsBuf, 
                                        &count, 
                                        CV_CALIB_CB_ADAPTIVE_THRESH);

        if (found)
        {
            // improve the found corners' coordinate accuracy
            cvFindCornerSubPix(m_pImg, 
                               m_pImagePointsBuf, 
                               count, 
                               cvSize(11,11), 
                               cvSize(-1,-1), 
                               cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

            cvDrawChessboardCorners(m_pImg, 
                                    m_BoardSize, 
                                    m_pImagePointsBuf, 
                                    count, 
                                    found);
        }

        if (found)
            m_CaptureDelay++;

        if (found && (m_CaptureDelay>2))
        {
            // Add the image points
            cvSeqPush(m_pImagePointsSeq, m_pImagePointsBuf);

            // Do we have enough images?
            if((unsigned)m_pImagePointsSeq->total >= (unsigned)m_ImageCount )
            {
                // Do the calibration part
                cvReleaseMat(&m_pExtrParamsMatrix);
                cvReleaseMat(&m_pReprojErrsMatrix);

                int code = RunCalibration(m_pImagePointsSeq, 
                                          m_ImgSize, 
                                          m_BoardSize,
                                          m_SquareSize, 
                                          m_AspectRatio, 
                                          m_Flags, 
                                          &m_CameraMatrix, 
                                          &m_DistCoeffsMatrix, 
                                          &m_pExtrParamsMatrix,
                                          &m_pReprojErrsMatrix, 
                                          &m_AvgReprojRrr);

                // Did the calibration succeed?
                if(code)
                {
                    m_bCalibrated = true;
                    m_bCalibratedChanged = true;

                    // Here we get the optical center and the focal length values from the intrinsic/camera matric
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_CameraMatrix, double, 0,0));
                    SetDlgItemText(IDC_FOCAL_LENGTH_X_EDIT, valueString);
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_CameraMatrix, double, 1,1));
                    SetDlgItemText(IDC_FOCAL_LENGTH_Y_EDIT, valueString);
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_CameraMatrix, double, 0,2));
                    SetDlgItemText(IDC_OPTICAL_CENTER_X_EDIT, valueString);
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_CameraMatrix, double, 1,2));
                    SetDlgItemText(IDC_OPTICAL_CENTER_Y_EDIT, valueString);

                    // Here we get the lens distortion coefficients (Notice: they are packed as: k1,k2,p1,p2,k3)
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_DistCoeffsMatrix, double, 0,0));
                    SetDlgItemText(IDC_K1_EDIT, valueString);
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_DistCoeffsMatrix, double, 0,1));
                    SetDlgItemText(IDC_K2_EDIT, valueString);
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_DistCoeffsMatrix, double, 0,2));
                    SetDlgItemText(IDC_P1_EDIT, valueString);
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_DistCoeffsMatrix, double, 0,3));
                    SetDlgItemText(IDC_P2_EDIT, valueString);
                    valueString.Format(_T("%0.5f"), CV_MAT_ELEM(m_DistCoeffsMatrix, double, 0,4));
                    SetDlgItemText(IDC_K3_EDIT, valueString);

                    // Prepare for the Undistort
                    cvInitUndistortMap(&m_CameraMatrix, &m_DistCoeffsMatrix, m_pUndistortMapX, m_pUndistortMapY);
                }
            }

            // Blink the image so we signal that we successfully took a picture
            cvNot(m_pImg, m_pImg);
            m_CaptureDelay = 0;
        }
    }

    EnterCriticalSection(&m_CriticalSection);
    if (m_bCalibrated && m_bSaveSettings)
    {
        // save camera parameters in any case, to catch Inf's/NaN's
        SaveCameraParams( "Settings.yml", 
            m_ImageCount, 
            m_ImgSize, 
            m_BoardSize, 
            m_SquareSize, 
            m_AspectRatio, 
            m_Flags, 
            &m_CameraMatrix, 
            &m_DistCoeffsMatrix, 
            m_pExtrParamsMatrix,
            m_pImagePointsSeq, 
            m_pReprojErrsMatrix, 
            m_AvgReprojRrr);
        m_bSaveSettings = false;
        m_bSettingsSaved = true;
    }
    LeaveCriticalSection(&m_CriticalSection);

    // Display the original image in the "Source" window
//    if (!m_bCalibrated)
        cvShowImage("Source", m_pImg);

    if (m_bCalibrated)
    {
        IplImage* t0 = cvCloneImage( m_pImg );
        cvRemap(t0, m_pImg, m_pUndistortMapX, m_pUndistortMapY);

        // Display the result
        cvShowImage("Output", m_pImg);

        EnterCriticalSection(&m_CriticalSection);
        if (m_bSaveImages)
        {
            cvSaveImage("SourceImage.tif", t0);
            cvSaveImage("OutputImage.tif", m_pImg);
            m_bSaveImages = false;
            m_bImagesSaved = true;
        }
        LeaveCriticalSection(&m_CriticalSection);
        cvReleaseImage( &t0 );
    }

}
//--------------------------------------------------------------------------------------------------
// OnBnClickedStop
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::OnBnClickedStop()
{
    J_STATUS_TYPE retval;

    // Stop Acquision
    if (m_hCam) {
        retval = J_Camera_ExecuteCommand(m_hCam, NODE_NAME_ACQSTOP);
    }

    if(m_hThread)
    {
        // Close stream
        retval = J_Image_CloseStream(m_hThread);
        m_hThread = NULL;
        TRACE("Closed stream\n");
    }

    cvDestroyWindow("Source");
    cvDestroyWindow("Output");
    if (m_pImg != NULL)
    {
        cvReleaseImage(&m_pImg);
        m_pImg = NULL;
    }

    if(m_pStorage)
    {
        cvReleaseMemStorage(&m_pStorage);
        m_pStorage = 0;
    }

    m_bAcquisitionRunning = false;
    EnableControls();
    KillTimer(1);
}

//--------------------------------------------------------------------------------------------------
// InitializeControls
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::InitializeControls()
{
    J_STATUS_TYPE   retval;
    NODE_HANDLE hNode;
    int64_t int64Val;

    CSliderCtrl* pSCtrl;

    //- Width ------------------------------------------------

    // Get SliderCtrl for Width
    pSCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER_WIDTH);

    // Get Width Node
    retval = J_Camera_GetNodeByName(m_hCam, NODE_NAME_WIDTH, &hNode);
    if (retval == J_ST_SUCCESS)
    {
        // Get/Set Min
        retval = J_Node_GetMinInt64(hNode, &int64Val);
        pSCtrl->SetRangeMin((int)int64Val, TRUE);

        // Get/Set Max
        retval = J_Node_GetMaxInt64(hNode, &int64Val);
        pSCtrl->SetRangeMax((int)int64Val, TRUE);

        // Get/Set Value
        retval = J_Node_GetValueInt64(hNode, FALSE, &int64Val);
        pSCtrl->SetPos((int)int64Val);

        SetDlgItemInt(IDC_WIDTH, (int)int64Val);
    }
    else
    {
        pSCtrl->ShowWindow(SW_HIDE);
        GetDlgItem(IDC_LBL_WIDTH)->ShowWindow(SW_HIDE);
        GetDlgItem(IDC_WIDTH)->ShowWindow(SW_HIDE);
    }

    //- Height -----------------------------------------------

    // Get SliderCtrl for Height
    pSCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER_HEIGHT);

    // Get Height Node
    retval = J_Camera_GetNodeByName(m_hCam, NODE_NAME_HEIGHT, &hNode);
    if (retval == J_ST_SUCCESS)
    {
        // Get/Set Min
        retval = J_Node_GetMinInt64(hNode, &int64Val);
        pSCtrl->SetRangeMin((int)int64Val, TRUE);

        // Get/Set Max
        retval = J_Node_GetMaxInt64(hNode, &int64Val);
        pSCtrl->SetRangeMax((int)int64Val, TRUE);

        // Get/Set Value
        retval = J_Node_GetValueInt64(hNode, FALSE, &int64Val);
        pSCtrl->SetPos((int)int64Val);

        SetDlgItemInt(IDC_HEIGHT, (int)int64Val);
    }
    else
    {
        pSCtrl->ShowWindow(SW_HIDE);
        GetDlgItem(IDC_LBL_HEIGHT)->ShowWindow(SW_HIDE);
        GetDlgItem(IDC_HEIGHT)->ShowWindow(SW_HIDE);
    }

    //- Gain -----------------------------------------------

    // Get SliderCtrl for Gain
    pSCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER_GAIN);

    // Get Gain Node
    retval = J_Camera_GetNodeByName(m_hCam, NODE_NAME_GAIN, &hNode);
    if (retval == J_ST_SUCCESS)
    {
        // Get/Set Min
        retval = J_Node_GetMinInt64(hNode, &int64Val);
        pSCtrl->SetRangeMin((int)int64Val, TRUE);

        // Get/Set Max
        retval = J_Node_GetMaxInt64(hNode, &int64Val);
        pSCtrl->SetRangeMax((int)int64Val, TRUE);

        // Get/Set Value
        retval = J_Node_GetValueInt64(hNode, FALSE, &int64Val);
        pSCtrl->SetPos((int)int64Val);

        SetDlgItemInt(IDC_GAIN, (int)int64Val);
    }
    else
    {
        pSCtrl->ShowWindow(SW_HIDE);
        GetDlgItem(IDC_LBL_GAIN)->ShowWindow(SW_HIDE);
        GetDlgItem(IDC_GAIN)->ShowWindow(SW_HIDE);
    }

    CString calibProcedure = _T("The camera calibration process will automatically start when the image acquisition is started and a Chess Board pattern with the specified black and white rectangular squares is put in front of the camera. After the specified number of sample images have been captured then the calibration is performed and the results are displayed.\nDuring the calibration process all the Chess Board inner corners will be marked whenever they are detected.");
    SetDlgItemText(IDC_CALIB_PROCEDURE_STATIC, calibProcedure);

    ((CSpinButtonCtrl*)GetDlgItem(IDC_CHESS_ROWS_SPIN))->SetPos(m_BoardSize.width+1);
    ((CSpinButtonCtrl*)GetDlgItem(IDC_CHESS_COLS_SPIN))->SetPos(m_BoardSize.height+1);
    ((CSpinButtonCtrl*)GetDlgItem(IDC_IMAGE_COUNT_SPIN))->SetPos(m_ImageCount);
}

//--------------------------------------------------------------------------------------------------
// EnableControls
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::EnableControls()
{
    GetDlgItem(IDC_START)->EnableWindow(m_bCameraOpen ? (m_bAcquisitionRunning ? FALSE : TRUE) : FALSE);
    GetDlgItem(IDC_STOP)->EnableWindow(m_bCameraOpen ? (m_bAcquisitionRunning ? TRUE : FALSE) : FALSE);
    GetDlgItem(IDC_SLIDER_WIDTH)->EnableWindow(m_bCameraOpen ? (m_bAcquisitionRunning ? FALSE : TRUE) : FALSE);
    GetDlgItem(IDC_WIDTH)->EnableWindow(m_bCameraOpen ? (m_bAcquisitionRunning ? FALSE : TRUE) : FALSE);
    GetDlgItem(IDC_SLIDER_HEIGHT)->EnableWindow(m_bCameraOpen ? (m_bAcquisitionRunning ? FALSE : TRUE) : FALSE);
    GetDlgItem(IDC_HEIGHT)->EnableWindow(m_bCameraOpen ? (m_bAcquisitionRunning ? FALSE : TRUE) : FALSE);
    GetDlgItem(IDC_SLIDER_GAIN)->EnableWindow(m_bCameraOpen ? TRUE : FALSE);
    GetDlgItem(IDC_GAIN)->EnableWindow(m_bCameraOpen ? TRUE : FALSE);
    GetDlgItem(IDC_SAVE_IMAGE_BUTTON)->EnableWindow(m_bCalibrated ? TRUE : FALSE);
    GetDlgItem(IDC_SAVE_SETTINGS_BUTTON)->EnableWindow(m_bCalibrated ? TRUE : FALSE);
    GetDlgItem(IDC_CHESS_ROWS_SPIN)->EnableWindow(m_bAcquisitionRunning ? FALSE : TRUE);
    GetDlgItem(IDC_CHESS_COLS_SPIN)->EnableWindow(m_bAcquisitionRunning ? FALSE : TRUE);
    GetDlgItem(IDC_IMAGE_COUNT_SPIN)->EnableWindow(m_bAcquisitionRunning ? FALSE : TRUE);
}

//--------------------------------------------------------------------------------------------------
// OnHScroll
//--------------------------------------------------------------------------------------------------
void COpenCVCameraCalibrationSampleDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
    // Get SliderCtrl for Width
    CSliderCtrl* pSCtrl;
    int iPos;
    J_STATUS_TYPE   retval;
    int64_t int64Val;

    //- Width ------------------------------------------------

    // Get SliderCtrl for Width
    pSCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER_WIDTH);
    if (pSCtrl == (CSliderCtrl*)pScrollBar) {

        iPos = pSCtrl->GetPos();
        int64Val = (int64_t)iPos;

        // Get Width Node
        retval = J_Camera_SetValueInt64(m_hCam, NODE_NAME_WIDTH, int64Val);

        SetDlgItemInt(IDC_WIDTH, iPos);
    }

    //- Height -----------------------------------------------

    // Get SliderCtrl for Height
    pSCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER_HEIGHT);
    if (pSCtrl == (CSliderCtrl*)pScrollBar) {

        iPos = pSCtrl->GetPos();
        int64Val = (int64_t)iPos;

        // Get Height Node
        retval = J_Camera_SetValueInt64(m_hCam, NODE_NAME_HEIGHT, int64Val);

        SetDlgItemInt(IDC_HEIGHT, iPos);
    }

    //- Gain -----------------------------------------------

    // Get SliderCtrl for Gain
    pSCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER_GAIN);
    if (pSCtrl == (CSliderCtrl*)pScrollBar) {

        iPos = pSCtrl->GetPos();
        int64Val = (int64_t)iPos;

        // Get Height Node
        retval = J_Camera_SetValueInt64(m_hCam, NODE_NAME_GAIN, int64Val);

        SetDlgItemInt(IDC_GAIN, iPos);
    }

    CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
}

double COpenCVCameraCalibrationSampleDlg::ComputeReprojectionError( const CvMat* object_points,
        const CvMat* rot_vects, const CvMat* trans_vects,
        const CvMat* camera_matrix, const CvMat* dist_coeffs,
        const CvMat* image_points, const CvMat* point_counts,
        CvMat* per_view_errors )
{
    CvMat* image_points2 = cvCreateMat( image_points->rows,
        image_points->cols, image_points->type );
    int i, image_count = rot_vects->rows, points_so_far = 0;
    double total_err = 0, err;
    
    for( i = 0; i < image_count; i++ )
    {
        CvMat object_points_i, image_points_i, image_points2_i;
        int point_count = point_counts->data.i[i];
        CvMat rot_vect, trans_vect;

        cvGetCols( object_points, &object_points_i,
            points_so_far, points_so_far + point_count );
        cvGetCols( image_points, &image_points_i,
            points_so_far, points_so_far + point_count );
        cvGetCols( image_points2, &image_points2_i,
            points_so_far, points_so_far + point_count );
        points_so_far += point_count;

        cvGetRow( rot_vects, &rot_vect, i );
        cvGetRow( trans_vects, &trans_vect, i );

        cvProjectPoints2( &object_points_i, &rot_vect, &trans_vect,
                          camera_matrix, dist_coeffs, &image_points2_i,
                          0, 0, 0, 0, 0 );
        err = cvNorm( &image_points_i, &image_points2_i, CV_L1 );
        if( per_view_errors )
            per_view_errors->data.db[i] = err/point_count;
        total_err += err;
    }
    
    cvReleaseMat( &image_points2 );
    return total_err/points_so_far;
}


int COpenCVCameraCalibrationSampleDlg::RunCalibration( CvSeq* image_points_seq, CvSize img_size, CvSize board_size,
                     float square_size, float aspect_ratio, int flags,
                     CvMat* camera_matrix, CvMat* dist_coeffs, CvMat** extr_params,
                     CvMat** reproj_errs, double* avg_reproj_err )
{
    int code;
    int image_count = image_points_seq->total;
    int point_count = board_size.width*board_size.height;
    CvMat* image_points = cvCreateMat( 1, image_count*point_count, CV_32FC2 );
    CvMat* object_points = cvCreateMat( 1, image_count*point_count, CV_32FC3 );
    CvMat* point_counts = cvCreateMat( 1, image_count, CV_32SC1 );
    CvMat rot_vects, trans_vects;
    int i, j, k;
    CvSeqReader reader;
    cvStartReadSeq( image_points_seq, &reader );

    // initialize arrays of points
    for( i = 0; i < image_count; i++ )
    {
        CvPoint2D32f* src_img_pt = (CvPoint2D32f*)reader.ptr;
        CvPoint2D32f* dst_img_pt = ((CvPoint2D32f*)image_points->data.fl) + i*point_count;
        CvPoint3D32f* obj_pt = ((CvPoint3D32f*)object_points->data.fl) + i*point_count;

        for( j = 0; j < board_size.height; j++ )
            for( k = 0; k < board_size.width; k++ )
            {
                *obj_pt++ = cvPoint3D32f(j*square_size, k*square_size, 0);
                *dst_img_pt++ = *src_img_pt++;
            }
        CV_NEXT_SEQ_ELEM( image_points_seq->elem_size, reader );
    }

    cvSet( point_counts, cvScalar(point_count) );

    *extr_params = cvCreateMat( image_count, 6, CV_32FC1 );
    cvGetCols( *extr_params, &rot_vects, 0, 3 );
    cvGetCols( *extr_params, &trans_vects, 3, 6 );

    cvZero( camera_matrix );
    cvZero( dist_coeffs );

    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
    {
        camera_matrix->data.db[0] = aspect_ratio;
        camera_matrix->data.db[4] = 1.;
    }

    cvCalibrateCamera2( object_points, image_points, point_counts,
                        img_size, camera_matrix, dist_coeffs,
                        &rot_vects, &trans_vects, flags );

    code = cvCheckArr( camera_matrix, CV_CHECK_QUIET ) &&
        cvCheckArr( dist_coeffs, CV_CHECK_QUIET ) &&
        cvCheckArr( *extr_params, CV_CHECK_QUIET );

    *reproj_errs = cvCreateMat( 1, image_count, CV_64FC1 );
    *avg_reproj_err =
        ComputeReprojectionError( object_points, &rot_vects, &trans_vects,
            camera_matrix, dist_coeffs, image_points, point_counts, *reproj_errs );

    cvReleaseMat( &object_points );
    cvReleaseMat( &image_points );
    cvReleaseMat( &point_counts );

    return code;
}


void COpenCVCameraCalibrationSampleDlg::SaveCameraParams( const char* out_filename, int image_count, CvSize img_size,
                         CvSize board_size, float square_size,
                         float aspect_ratio, int flags,
                         const CvMat* camera_matrix, CvMat* dist_coeffs,
                         const CvMat* extr_params, const CvSeq* image_points_seq,
                         const CvMat* reproj_errs, double avg_reproj_err )
{
    CvFileStorage* fs = cvOpenFileStorage( out_filename, 0, CV_STORAGE_WRITE );
    
    time_t t;
    time( &t );
    struct tm t2; 
    errno_t error = localtime_s(&t2, &t );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", &t2 );

    cvWriteString( fs, "calibration_time", buf );
    
    cvWriteInt( fs, "image_count", image_count );
    cvWriteInt( fs, "image_width", img_size.width );
    cvWriteInt( fs, "image_height", img_size.height );
    cvWriteInt( fs, "board_width", board_size.width );
    cvWriteInt( fs, "board_height", board_size.height );
    cvWriteReal( fs, "square_size", square_size );
    
    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        cvWriteReal( fs, "aspect_ratio", aspect_ratio );

    if( flags != 0 )
    {
        sprintf_s( buf, sizeof(buf), "flags: %s%s%s%s",
            flags & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspect_ratio" : "",
            flags & CV_CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        cvWriteComment( fs, buf, 0 );
    }
    
    cvWriteInt( fs, "flags", flags );

    cvWrite( fs, "camera_matrix", camera_matrix );
    cvWrite( fs, "distortion_coefficients", dist_coeffs );

    cvWriteReal( fs, "avg_reprojection_error", avg_reproj_err );
    if( reproj_errs )
        cvWrite( fs, "per_view_reprojection_errors", reproj_errs );

    if( extr_params )
    {
        cvWriteComment( fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        cvWrite( fs, "extrinsic_parameters", extr_params );
    }

    if( image_points_seq )
    {
        cvWriteComment( fs, "the array of board corners projections used for calibration", 0 );
        assert( image_points_seq->total == image_count );
        CvMat* image_points = cvCreateMat( 1, image_count*board_size.width*board_size.height, CV_32FC2 );
        cvCvtSeqToArray( image_points_seq, image_points->data.fl );

        cvWrite( fs, "image_points", image_points );
        cvReleaseMat( &image_points );
    }

    cvReleaseFileStorage( &fs );
}

void COpenCVCameraCalibrationSampleDlg::OnBnClickedCalibrateButton()
{
    if (m_bCalibrated)
    {
        if(m_pStorage)
        {
            cvReleaseMemStorage(&m_pStorage);
            m_pStorage = 0;
        }

        int elem_size = m_BoardSize.width*m_BoardSize.height*sizeof(CvPoint2D32f);
        if (m_pStorage == 0)
        {
            m_pStorage = cvCreateMemStorage(MAX( elem_size*4, 1 << 16 ));
            m_pImagePointsBuf = (CvPoint2D32f*)cvAlloc( elem_size );
            m_pImagePointsSeq = cvCreateSeq( 0, sizeof(CvSeq), elem_size, m_pStorage);
        }

        m_bCalibrated = false;
        m_bCalibratedChanged = true;
        EnableControls();
    }
}

void COpenCVCameraCalibrationSampleDlg::OnBnClickedSaveImageButton()
{
    EnterCriticalSection(&m_CriticalSection);
    m_bSaveImages = true;
    LeaveCriticalSection(&m_CriticalSection);
}

void COpenCVCameraCalibrationSampleDlg::OnBnClickedSaveSettingsButton()
{
    EnterCriticalSection(&m_CriticalSection);
    m_bSaveSettings = true;
    LeaveCriticalSection(&m_CriticalSection);
}

void COpenCVCameraCalibrationSampleDlg::OnDeltaposChessRowsSpin(NMHDR *pNMHDR, LRESULT *pResult)
{
    LPNMUPDOWN pNMUpDown = reinterpret_cast<LPNMUPDOWN>(pNMHDR);
    m_BoardSize.width = pNMUpDown->iPos-1;

    *pResult = 0;
}

void COpenCVCameraCalibrationSampleDlg::OnDeltaposChessColsSpin(NMHDR *pNMHDR, LRESULT *pResult)
{
    LPNMUPDOWN pNMUpDown = reinterpret_cast<LPNMUPDOWN>(pNMHDR);
    m_BoardSize.height = pNMUpDown->iPos-1;
    *pResult = 0;
}

void COpenCVCameraCalibrationSampleDlg::OnDeltaposImageCountSpin(NMHDR *pNMHDR, LRESULT *pResult)
{
    LPNMUPDOWN pNMUpDown = reinterpret_cast<LPNMUPDOWN>(pNMHDR);
    m_ImageCount = pNMUpDown->iPos;
    *pResult = 0;
}
