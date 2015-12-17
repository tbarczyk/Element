// OpenCVCameraCalibrationSampleDlg.h : header file
//

#pragma once

#include <Jai_Factory.h>

#include "Element.h"
// OpenCV includes required
#include <opencv2/opencv.hpp>
//#include <opencv2/legacy/compat.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#define NODE_NAME_WIDTH         (int8_t*)"Width"
#define NODE_NAME_HEIGHT        (int8_t*)"Height"
#define NODE_NAME_PIXELFORMAT   (int8_t*)"PixelFormat"
#define NODE_NAME_GAIN          (int8_t*)"GainRaw"
#define NODE_NAME_ACQSTART      (int8_t*)"AcquisitionStart"
#define NODE_NAME_ACQSTOP       (int8_t*)"AcquisitionStop"

// CVCSample1Dlg dialog
class COpenCVCameraCalibrationSampleDlg : public CDialog
{
    // Construction
public:
    COpenCVCameraCalibrationSampleDlg(CWnd* pParent = NULL);	// standard constructor
    // Dialog Data
    enum { IDD = IDD_OPENCVCAMERACALIBRATIONSAMPLE_DIALOG };

protected:
    virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support

    // Implementation
public:
    FACTORY_HANDLE  m_hFactory;     // Factory Handle
    CAM_HANDLE      m_hCam;         // Camera Handle
    THRD_HANDLE     m_hThread;      // Acquisition Thread Handle
    int8_t          m_sCameraId[J_CAMERA_ID_SIZE];    // Camera ID
    IplImage*       m_pImg;        // OpenCV Images
    bool            m_bCameraOpen;
    bool            m_bAcquisitionRunning;
    CvSize          m_BoardSize;
    CvSize          m_ImgSize;
    CvPoint2D32f*   m_pImagePointsBuf;
    CvSeq*          m_pImagePointsSeq;
    double          m_Camera[9];
    double          m_DistCoeffs[5];
    CvMat           m_CameraMatrix;
    CvMat           m_DistCoeffsMatrix;
    CvMat*          m_pExtrParamsMatrix;
    CvMat*          m_pReprojErrsMatrix;
    double          m_AvgReprojRrr;
    bool            m_bCalibrated;
    bool            m_bCalibratedChanged;
    int             m_ImageCount;
    float           m_SquareSize;
    float           m_AspectRatio;
    int             m_Flags;
    int             m_CaptureDelay;
    bool            m_bSaveImages;
    bool            m_bImagesSaved;
    bool            m_bSaveSettings;
    bool            m_bSettingsSaved;
	VIEW_HANDLE     m_hView;
    CRITICAL_SECTION m_CriticalSection;             // Critical section used for protecting the measured time value so it can be displayed
    CvMemStorage*   m_pStorage;
    IplImage*       m_pUndistortMapX;
    IplImage*       m_pUndistortMapY;

    BOOL OpenFactoryAndCamera();
    void CloseFactoryAndCamera();
    void StreamCBFunc(J_tIMAGE_INFO * pAqImageInfo);
    void InitializeControls();
    void EnableControls();

    double ComputeReprojectionError( const CvMat* object_points,
        const CvMat* rot_vects, const CvMat* trans_vects,
        const CvMat* camera_matrix, const CvMat* dist_coeffs,
        const CvMat* image_points, const CvMat* point_counts,
        CvMat* per_view_errors );

    int RunCalibration( CvSeq* image_points_seq, CvSize img_size, CvSize board_size,
                     float square_size, float aspect_ratio, int flags,
                     CvMat* camera_matrix, CvMat* dist_coeffs, CvMat** extr_params,
                     CvMat** reproj_errs, double* avg_reproj_err );
    void SaveCameraParams( const char* out_filename, int image_count, CvSize img_size,
                         CvSize board_size, float square_size,
                         float aspect_ratio, int flags,
                         const CvMat* camera_matrix, CvMat* dist_coeffs,
                         const CvMat* extr_params, const CvSeq* image_points_seq,
                         const CvMat* reproj_errs, double avg_reproj_err );


protected:
    HICON m_hIcon;

    // Generated message map functions
    virtual BOOL OnInitDialog();
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    DECLARE_MESSAGE_MAP()
public:
    afx_msg void OnDestroy();
public:
    afx_msg void OnBnClickedStart();
public:
    afx_msg void OnBnClickedStop();
public:
    afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
    afx_msg void OnTimer(UINT_PTR nIDEvent);
    afx_msg void OnBnClickedCalibrateButton();
    afx_msg void OnBnClickedSaveImageButton();
    afx_msg void OnBnClickedSaveSettingsButton();
    afx_msg void OnDeltaposChessRowsSpin(NMHDR *pNMHDR, LRESULT *pResult);
    afx_msg void OnDeltaposChessColsSpin(NMHDR *pNMHDR, LRESULT *pResult);
    afx_msg void OnDeltaposImageCountSpin(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnBnClickedFilescalib();
	afx_msg void OnBnClickedBtn();
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedStreambutton();
	afx_msg void OnBnClickedOclbtn();
};

