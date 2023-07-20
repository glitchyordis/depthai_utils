import depthai as dai
import time
import yaml
import PySimpleGUI as gui
import pathlib
from typing import Dict, Tuple, Union, List, Optional, Any
from attrs import define, field
import sys
import cv2
import numpy as np
from numpy.typing import NDArray
import datetime
import math
import pandas as pd
from copy import deepcopy


from .camera_utils import PredefinedAFRegion
from .gui_windows import GUIWindows
from .polaris_logger import logger, log_level_keys
from . import POLARIS_CONSTANTS as PO_CONST
from . import polaris_utils as po_utils
from . import camera_utils as cam_utils

with open(PO_CONST.SES_SETTING_YAML, "r") as stream:
        SES_SETTING = list(yaml.safe_load_all(stream))[0]
devordebug = po_utils.devordebug(SES_SETTING)

CAM_AF_OPT = PO_CONST.CAM_AF_OPT

CAM_STREAM_NAMES = PO_CONST.CAM_PARAM["stream_names"]

THelper = po_utils.TextHelper()

GET_DISPA_FRAME : bool = SES_SETTING["cam"]["get_disparity_frame"]
GET_SYNC_FRAME : bool  = SES_SETTING["cam"]["get_synced_frame"]
COBOT_ENABLED : bool = SES_SETTING["cobot"]["enabled"][0]


class HostSync_ts:
    """
    a timestamp based synching of Oak-D frames
    """

    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if name not in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({'msg': msg})
        # Try finding synced msgs
        ts = msg.getTimestamp()
        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                time_diff = abs(obj['msg'].getTimestamp() - ts)
                # 20ms since we add rgb/depth frames at 30FPS => 33ms. If
                # time difference is below 20ms, it's considered as synced
                if time_diff < datetime.timedelta(milliseconds=33):
                    synced[name] = obj['msg']
                    # print(f"{name}: {i}/{len(arr)}")
                    break
        # If there are 2 ( color, depth) synced msgs, remove all old msgs & return synced msgs
        if len(synced) == 2: # 
            def remove(t1, t2):
                return datetime.timedelta(milliseconds=500) < abs(t1 - t2) 
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if remove(obj['msg'].getTimestamp(), ts):
                        arr.remove(obj)
                    else: break
            return synced
        return False

class Camera():
    """
    ! remember to pass in gui_wins after it has been initialized

    OAK-D PRO
    
    color cam resolution: 
        12MP 4056x3040


    Note
        Syncframes have a delay and at the start of displaying a synced frame, 
            if a cv window was not created before hand,
        cv2.getWindowProperty(cv2_windowname, cv2.WND_PROP_VISIBLE) < 1: 
            will identify that window as does not exist
    """

    def __init__(self) -> None:
        self.gui_wins : GUIWindows =  GUIWindows() # polars gui class

        self.CAM_PARAM = PO_CONST.CAM_PARAM

        with open(PO_CONST.SES_SETTING_YAML, "r") as stream:
            self.SES_SETTING = list(yaml.safe_load_all(stream))[0]

        self.SES_CAM_SETTING = self.SES_SETTING["cam"]

        # camera status
        self.camera_failed : bool = True
        self.live_stream: bool = False

        # camera properties
        self.camRgb : Optional[dai.node.ColorCamera] = None
        self.oakMxid, self.device =  None, None
        self.pipeline : dai.Pipeline = dai.Pipeline()
        self.dispaQ : Optional[dai.DataOutputQueue] = None
        self.qSysInfo : Optional[dai.DataOutputQueue] = None
        self.controlQueue : Optional[dai.DataInputQueue] = None
        self.rgbQueue : Optional[dai.DataInputQueue] = None # this is only used for manual polaris/ dataset taker
        self.control : Optional[str] = None  # contorl for cam such as AE COMP
        self.cam_outq : List[dai.DataOutputQueue] = []
        self.lensPos : int = 0
        self.ctrl_param = PO_CONST.CamControlParam()

        self.calibData = None
        self.maxDisparity = None

        # camera frames
        self.bgrData : Optional[dai.ImgFrame] = None
        self.bgrframe : NDArray[np.uint8] = np.zeros((0, 0, 3)) # ! dont use == np.zeros((0, 0, 3)) to test, instead use .shape == np.zeros((0, 0, 3)).shape
        self.resized_bgr : NDArray[np.uint8] = np.zeros((0, 0, 3))
        self.dispaFrame = None
        self.static_depthmsg = None
        self.depthFrame = None
        
        self.sync = HostSync_ts()
        self.synced_msgs : Union[bool, Dict] = False

        # AF
        self.wait_AF_duration = 3

        # for displaying depth at mouse coordinates on frame
        self.mouse_point = (0,0)

    def find_oakMxid(self): # @ last update 05 jun 2023
        oakMxid, calibData, device = None, None, None

        for i in dai.Device.getAllAvailableDevices():
            oakMxid = i.mxid

            device = dai.Device(oakMxid, maxUsbSpeed=dai.UsbSpeed.SUPER)
            
            # device.setLogLevel(dai.LogLevel.INFO)
            # device.setLogOutputLevel(dai.LogLevel.INFO)

            calibData = device.readCalibration()

            break
            
        logger.info(f"Found oakMxid: {oakMxid}")

        return oakMxid, device, calibData
    
    def create_dataset_collector_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setXLinkChunkSize(0)

        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

        # imu = pipeline.create(dai.node.IMU)
        
        controlIn = pipeline.create(dai.node.XLinkIn)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        imu_out = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName(self.CAM_PARAM["stream_names"]["rgb"])
        imu_out.setStreamName("imu")
        controlIn.setStreamName(self.CAM_PARAM["stream_names"]['ctrl'])

        camRgb.isp.link(xoutRgb.input)
        camRgb.initialControl.setAutoFocusMode(CAM_AF_OPT['CONT_VID'])
        controlIn.out.link(camRgb.inputControl) 

        # imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
        # # it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
        # # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
        # imu.setBatchReportThreshold(1)
        # # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
        # # if lower or equal to batchReportThreshold then the sending is always blocking on device
        # # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
        # imu.setMaxBatchReports(10)
        # imu.out.link(imu_out.input)

        self.pipeline = pipeline
        self.camRgb = camRgb

    def create_pipeline(self): # @ last update 05 jun 2023

        pipeline = dai.Pipeline()
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        
        # control
        controlIn = pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName(self.CAM_PARAM["stream_names"]['ctrl'])
        controlIn.out.link(camRgb.inputControl) 

        if not self.SES_CAM_SETTING["get_synced_frame"]:

            # cam feed
            xout_isp = pipeline.create(dai.node.XLinkOut)
            xout_isp.setStreamName(self.CAM_PARAM["stream_names"]['isp'])
            camRgb.isp.link(xout_isp.input)

        else:
            pipeline.setXLinkChunkSize(0)

            # Define sources and outputs
            monoLeft = pipeline.create(dai.node.MonoCamera)
            monoRight = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)

            xoutRgb = pipeline.create(dai.node.XLinkOut)
            xoutDepth = pipeline.create(dai.node.XLinkOut) 

            xoutRgb.setStreamName(self.CAM_PARAM["stream_names"]["rgb"])
            xoutDepth.setStreamName(self.CAM_PARAM["stream_names"]["depth"])

            # Properties
            camRgb.setFps(self.CAM_PARAM['rgb']['fps'])
            # camRgb.setInterleaved(False) 
            # try:
            #     calibData = device.readCalibration2()
            #     lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            #     if lensPosition:
            #         print(f"tuned lens pos for depth: {lensPosition}")
            #         camRgb.initialControl.setManualFocus(lensPosition)
            #     else:
            #         gui.popup("No Calib data for lensposition")
            # except:
            #     raise

            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
            monoLeft.setFps(self.CAM_PARAM['monoL']['fps'])
            monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            monoRight.setFps(self.CAM_PARAM['monoR']['fps'])

            # Stereo config
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(True)

            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            # this is similar doing threshold (int) -> stereo_depth.initialConfig.setConfidenceThreshold(threshold)
            
            # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

            stereoConfig = stereo.initialConfig.get()
            stereoConfig.postProcessing.speckleFilter.enable = True
            stereoConfig.postProcessing.speckleFilter.speckleRange = 50
            # stereoConfig.postProcessing.temporalFilter.enable = True
            stereoConfig.postProcessing.spatialFilter.enable = True
            stereoConfig.postProcessing.spatialFilter.holeFillingRadius = 2
            stereoConfig.postProcessing.spatialFilter.numIterations = 1
            stereoConfig.postProcessing.thresholdFilter.minRange = self.CAM_PARAM["stereo"]["thresholdFilter"]["low"]
            stereoConfig.postProcessing.thresholdFilter.maxRange = self.CAM_PARAM["stereo"]["thresholdFilter"]["high"]
            stereoConfig.postProcessing.decimationFilter.decimationFactor = 2
            stereo.initialConfig.set(stereoConfig)

            # Align depth map to the perspective of RGB camera, on which inference is done
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setOutputSize(*self.CAM_PARAM["stereo"]["outputsz"])
            
            # Linking
            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)
            
            camRgb.isp.link(xoutRgb.input)
            stereo.depth.link(xoutDepth.input)

            if GET_DISPA_FRAME:
                xoutDisp = pipeline.create(dai.node.XLinkOut)
                xoutDisp.setStreamName(self.CAM_PARAM["stream_names"]["disparity"])
                stereo.disparity.link(xoutDisp.input)

            if self.SES_CAM_SETTING["get_oak_log"]:
                sysLog = pipeline.create(dai.node.SystemLogger)
                sysLog.setRate(0.2)  # 1 Hz

                linkOut = pipeline.create(dai.node.XLinkOut)
                linkOut.setStreamName(self.CAM_PARAM["stream_names"]["oak_log"])
                sysLog.out.link(linkOut.input)
            
            self.maxDisparity = stereo.initialConfig.getMaxDisparity()
        
        self.pipeline = pipeline
        self.camRgb = camRgb 
        
    def DispaFrame_getnshow(self): # @ last update 06 jun 2023
        if GET_DISPA_FRAME:
            dispaFrame = self.dispaQ.get().getFrame()
            dispaFrame = (dispaFrame * (255 / self.maxDisparity)).astype(np.uint8)
            dispaFrame = cv2.resize(dispaFrame, self.CAM_PARAM["stereo"]["outputsz"], interpolation=cv2.INTER_AREA)
            dispaFrame = cv2.applyColorMap(dispaFrame, cv2.COLORMAP_TURBO) 

            self.dispaFrame = dispaFrame
            cv2.imshow(PO_CONST.CV_WIN_NAME.DISPA, self.dispaFrame) 

    def start_dataset_collector_mode(self, save_path : pathlib.Path):
        """
        to use this externally we need the following code and POLARIS_CONSTANTS.py 
        and SESSION_SETTINGS.yaml

            guimaker = GUIMaker()
            gui_wins = GUIWindows()


            save_path = pathlib.Path.cwd()
            gui_wins.main_win = guimaker.make_dataset_collector_win(save_path)
            cam = Camera()
            cam.gui_wins = gui_wins

            cam.start_dataset_collector_mode(save_path)
        """
        self.oakMxid, self.device, self.calibData = self.find_oakMxid()

        if self.oakMxid is None:
                message = "No Camera detected. Exiting Script..."
                logger.critical("No Camera detected. Exiting Script...")
                gui.popup(message, title="Camera")
                sys.exit()
        else:
            self.camera_failed = False
            self.create_dataset_collector_pipeline()
            if not self.device.isPipelineRunning():
                self.device.startPipeline(self.pipeline)
            self.live_stream = True

            self.controlQueue = self.device.getInputQueue(self.CAM_PARAM["stream_names"]['ctrl'])
            self.rgbQueue = self.device.getOutputQueue(self.CAM_PARAM["stream_names"]['rgb'],
                                                        maxSize=1, blocking=False)
            
            self.start_dataset_collector_loop(save_path)

    def start(self): # @ last update 06 jun 2023

        if self.SES_CAM_SETTING["enable"]:
            # init cv window
            cv2.namedWindow(PO_CONST.CV_WIN_NAME.MAIN, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(PO_CONST.CV_WIN_NAME.MAIN, 0, 0)

            self.oakMxid, self.device, self.calibData = self.find_oakMxid()

            if self.oakMxid is None:
                message = "No Camera detected. Exiting Script..."
                logger.critical("No Camera detected. Exiting Script...")
                gui.popup(message, title="Camera")
                sys.exit()
            else:
                self.camera_failed = False
                self.create_pipeline()
                if not self.device.isPipelineRunning():
                    self.device.startPipeline(self.pipeline)
                self.live_stream = True

                self.controlQueue = self.device.getInputQueue(self.CAM_PARAM["stream_names"]['ctrl'])

                if GET_SYNC_FRAME:    
                    self.cam_outq = [self.device.getOutputQueue(name, 1, False) for name in [self.CAM_PARAM["stream_names"]['rgb'],
                                                                                            self.CAM_PARAM["stream_names"]['depth']]]
                    cv2.setMouseCallback(PO_CONST.CV_WIN_NAME.MAIN, self.show_distance_at_mousepoint)
            
            self.dispaQ : Optional[dai.DataOutputQueue] = self.set_dispaQ()
            self.qSysInfo : Optional[dai.DataOutputQueue] = self.set_qSysInfo()
        
    def start_dataset_collector_loop(self, save_path : pathlib.Path):
        while True:
            self.gui_wins.read_all()

            if self.gui_wins.check_exit():
                break
            
            self.get_bgrFrame_only()

            event = self.gui_wins.event

            if self.live_stream:
                imgbytes=cv2.imencode('.png', self.resized_bgr)[1].tobytes()
                self.gui_wins.update_valsNcolor(element_key = "-GRAPH-", data=imgbytes)

            if event in ["-AF_slider-", '-autofocus_combo-', ',','.']:
                self.setFocus_w_GUI()
            elif event == "-getLensPos-":
                self.gui_wins.out_msg2term(f"lensPos : {self.lensPos}")
            elif event == "-clearTerminal-":
                self.gui_wins.update_valsNcolor(element_key= '-OUTPUT-' + gui.WRITE_ONLY_KEY,
                                                value = "", force=("value"))
            elif event == '-Autoexposure_enable-':
                for gui_element in ["-exp_time-", "-iso-"]:
                    self.gui_wins.update_valsNcolor(element_key = gui_element, value="Auto")
                self.AutoExposure_enable()
            elif event == '-Autowhitebalance_enable-':
                self.gui_wins.update_valsNcolor(element_key = "-WB_temp-", value="Auto")
                self.AutoWhiteBalance_enable()
            elif event in ['i', 'o', 'k', 'l']:
                if event == 'i': 
                    self.ctrl_param.expTime -= self.ctrl_param.EXP_STEP
                if event == 'o': 
                    self.ctrl_param.expTime += self.ctrl_param.EXP_STEP
                if event == 'k': 
                    self.ctrl_param.sensIso -= self.ctrl_param.ISO_STEP
                if event == 'l': 
                    self.ctrl_param.sensIso += self.ctrl_param.ISO_STEP

                self.ctrl_param.expTime = po_utils.clamp(self.ctrl_param.expTime, 1, 33000)
                self.ctrl_param.sensIso =  po_utils.clamp(self.ctrl_param.sensIso, 100, 1600)

                self.gui_wins.update_valsNcolor(element_key= "-exp_time-", value=self.ctrl_param.expTime)
                self.gui_wins.update_valsNcolor(element_key= "-iso-", value=self.ctrl_param.sensIso)

                self.Exposure_set_manual()
            elif event in ['n', 'm']:
                if event == 'n': 
                    self.ctrl_param.wbManual -= self.ctrl_param.WB_STEP
                if event == 'm': 
                    self.ctrl_param.wbManual += self.ctrl_param.WB_STEP
                self.ctrl_param.wbManual = po_utils.clamp(self.ctrl_param.wbManual, 1000, 12000)
                
                self.gui_wins.update_valsNcolor(element_key="-WB_temp-", 
                                                value=self.ctrl_param.wbManual)
                self.WhiteBalance_set_manual()
            elif event == '1':
                self.ctrl_param.awb_lock = not(self.ctrl_param.awb_lock)
                self.gui_wins.update_valsNcolor(element_key="-AWB_lock-", 
                                                value = self.ctrl_param.awb_lock)
                self.AWBalance_lock_set()
            elif event == '2':
                self.ctrl_param.ae_lock = not(self.ctrl_param.ae_lock)
                self.gui_wins.update_valsNcolor(element_key = "-AE_lock-", 
                                                value = self.ctrl_param.ae_lock)
                self.AutoExposure_lock_set()
            elif len(event) == 1 and event in '34567890[]':
                if   event == '3': 
                    self.ctrl_param.control = 'awb_mode'
                elif event == '4': 
                    self.ctrl_param.control = 'ae_comp'
                elif event == '5': 
                    self.ctrl_param.control = 'anti_banding_mode'
                elif event == '6': 
                    self.ctrl_param.control = 'effect_mode'
                elif event == '7': 
                    self.ctrl_param.control = 'brightness'
                elif event == '8': 
                    self.ctrl_param.control = 'contrast'
                elif event == '9': 
                    self.ctrl_param.control = 'saturation'
                elif event == '0': 
                    self.ctrl_param.control = 'sharpness'
                elif event == '[': 
                    self.ctrl_param.control = 'luma_denoise'
                elif event ==']': 
                    self.ctrl_param.control = 'chroma_denoise'

                self.gui_wins.update_valsNcolor(element_key="-control-", 
                                                value = self.ctrl_param.control)
                self.gui_wins.main_win.refresh()

            elif event in ['-', '_', '+','=']:
                change = 0
                if event in ['-', '_']: 
                    change = -1
                if event in ['+', '=']: 
                    change = 1

                ctrl = dai.CameraControl()

                ele_key : str = ""
                ele_val : Any = None

                if self.ctrl_param.control == 'none':
                    print("Please select a control first using keys 3..9 0 [ ]")
                elif self.ctrl_param.control == 'ae_comp':
                    self.ctrl_param.ae_comp = po_utils.clamp(self.ctrl_param.ae_comp + change, -9, 9)
                    ele_key, ele_val = "-AE_comp-", self.ctrl_param.ae_comp
                    ctrl.setAutoExposureCompensation(self.ctrl_param.ae_comp)
                elif self.ctrl_param.control == 'anti_banding_mode':
                    abm = next(self.ctrl_param.anti_banding_mode)
                    ele_key, ele_val = "-Anti_banding-", abm
                    ctrl.setAntiBandingMode(abm)
                elif self.ctrl_param.control == 'awb_mode':
                    awb = next(self.ctrl_param.awb_mode)
                    ele_key, ele_val = "-AWB_mode-", awb
                    ctrl.setAutoWhiteBalanceMode(awb)
                elif self.ctrl_param.control == 'effect_mode':
                    eff = next(self.ctrl_param.effect_mode)
                    ele_key, ele_val = "-effect_mode-", eff
                    ctrl.setEffectMode(eff)
                elif self.ctrl_param.control == 'brightness':
                    self.ctrl_param.brightness = po_utils.clamp(self.ctrl_param.brightness + change, -10, 10)
                    ele_key, ele_val = "-brightness-", self.ctrl_param.brightness
                    ctrl.setBrightness(self.ctrl_param.brightness)
                elif self.ctrl_param.control == 'contrast':
                    self.ctrl_param.contrast = po_utils.clamp(self.ctrl_param.contrast + change, -10, 10)
                    ele_key, ele_val = "-contrast-" , self.ctrl_param.contrast
                    ctrl.setContrast(self.ctrl_param.contrast)
                elif self.ctrl_param.control == 'saturation':
                    self.ctrl_param.saturation = po_utils.clamp(self.ctrl_param.saturation + change, -10, 10)
                    ele_key, ele_val = "-saturation-", self.ctrl_param.saturation
                    ctrl.setSaturation(self.ctrl_param.saturation)
                elif self.ctrl_param.control == 'sharpness':
                    self.ctrl_param.sharpness = po_utils.clamp(self.ctrl_param.sharpness + change, 0, 4)
                    ele_key, ele_val = "-sharpness-" , self.ctrl_param.sharpness
                    ctrl.setSharpness(self.ctrl_param.sharpness)
                elif self.ctrl_param.control == 'luma_denoise':
                    self.ctrl_param.luma_denoise = po_utils.clamp(self.ctrl_param.luma_denoise + change, 0, 4)
                    ele_key, ele_val = "-luma_denoise-", self.ctrl_param.luma_denoise
                    ctrl.setLumaDenoise(self.ctrl_param.luma_denoise)
                elif self.ctrl_param.control == 'chroma_denoise':
                    self.ctrl_param.chroma_denoise = po_utils.clamp(self.ctrl_param.chroma_denoise + change, 0, 4)
                    ele_key, ele_val = "-chroma_denoise-", self.ctrl_param.chroma_denoise
                    ctrl.setChromaDenoise(self.ctrl_param.chroma_denoise)

                if ele_key:
                    self.gui_wins.update_valsNcolor(element_key= ele_key, 
                                                    value = ele_val)
                    self.send_ctrl_to_cam(ctrl)

            if event == "-capture-":
                if self.live_stream:
                    self.gui_wins.update_valsNcolor(element_key="cam_status",
                                                    value = "Image saved. Camera paused.", 
                                                    change_background=(True, "#ff9500"))
                    self.live_stream = not(self.live_stream)

                    date_now, time_now = po_utils.get_system_time()

                    folder = pathlib.Path(save_path/date_now/"dataset_raw")
                    folder.mkdir(parents=True,exist_ok=True)

                    try:
                        img_name = date_now + "_" + time_now
                        cv2.imwrite(str(folder/img_name) + ".png", self.bgrframe,
                                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        self.gui_wins.update_valsNcolor(element_key="cam_status",
                                                        value="Image saved. Camera paused.", 
                                                        change_background=(True, "#ff9500"))
                    except Exception as e:
                        self.gui_wins.update_valsNcolor(element_key="cam_status",
                                                        value=f"Error: {e}",
                                                        change_background=(True,"#ff3b30"))
                else:
                    self.live_stream = not(self.live_stream)
                    self.gui_wins.update_valsNcolor(element_key="cam_status",
                                                    value="Camera Live", 
                                                    change_background=(True, "#28cd41"))
            # if len(event) == 1:
            #     window["cam_status"].update(value='%s - %s' % (event, ord(event)))
            # if event not in ["__TIMEOUT__","-autofocus_DC_button-",'-autofocus_C_button-','-Autoexposure_enable-',
            # '-Autowhitebalance_enable-',',', '.','i', 'o', 'k', 'l','n', 'm']:
            #     window["cam_status"].update(event)

        self.gui_wins.main_win.close()
        if isinstance(self.device, dai.Device):
            self.device.close()

    # spatials
    def spatials_calc(self, data : pd.DataFrame, bboxScale : float = 0.5) -> Tuple[pd.DataFrame, Dict]:
                      
        """
        Function
            calculates the real-world x,y,z distance [mm] of the centroid of a ROI (from obj det) relative to OAK-D PRO color camera

        Params
            frame: bgrframe, 12MP
            frame_resized: a resized ver of `frame` for displaying to user
            data: pandas df of detected object from yolov5
        """

        def calc_angle(offset, HFOV):
            return math.atan(math.tan(HFOV / 2.0) * offset / (self.depthFrame.shape[1] / 2.0))
        
        def get_avgDepth(s_ymin, s_ymax, s_xmin, s_xmax):
            depthROI = self.depthFrame[s_ymin : s_ymax, s_xmin : s_xmax]
            inRange = (self.CAM_PARAM["stereo"]["thresholdFilter"]["low"] <= depthROI) & \
                (depthROI <= self.CAM_PARAM["stereo"]["thresholdFilter"]["high"])

            depthROI_inRange = depthROI[inRange] # this is already a 1D array
            averageDepth = np.mean(depthROI_inRange)

            """
            try:
                # actually there is really no need for this  since inRange already filter values within low and high. 
                # if we dont put low as 0 I think we good
                averageDepth_mad = 0 
                median = np.median(depthROI_inRange)
                mad = np.median(abs(depthROI_inRange - median))

                if mad == 0:
                    mad += 1e-9

                inRange_mad = (median-3*mad < depthROI_inRange) & (depthROI_inRange < median+3*mad)
                averageDepth_mad = np.mean(depthROI_inRange[inRange_mad])
            except Exception as e:
                print(f"Failed to calc MAD. Error: {e}")
            """

            try:
                averageDepth_mad = 0 
                median = np.median(depthROI_inRange)
                mad = np.median(abs(depthROI_inRange - median))

                if mad == 0:
                    mad += 1e-9

                inRange_mad = (median-3*mad < depthROI_inRange) & (depthROI_inRange < median+3*mad)
                averageDepth_mad = np.mean(depthROI_inRange[inRange_mad])

                logger.info(f"averageDepth: {averageDepth}; averageDepth_mad:{averageDepth_mad}")

                if averageDepth_mad!=0 and averageDepth_mad<averageDepth and abs(averageDepth_mad-averageDepth)>5:
                    logger.debug(f"averageDepth_mad: {averageDepth_mad}, averageDepth:{averageDepth}")
                    logger.debug(f"use averageDepth_mad instead. ") 

                    averageDepth = averageDepth_mad

            except Exception as e:
                message = f"Error \n{e}.\n{po_utils.get_exception_info()}"
                self.gui_wins.out_msg2term(message, lvl="red", popup=True, autolog=(True, log_level_keys.exception))

            return averageDepth
        
        def clamp_px(px):

            if bboxScale*px>=10:
                return bboxScale*px
            else:
                return px

        HFOV = np.deg2rad(self.calibData.getFov(dai.CameraBoardSocket(self.static_depthmsg.getInstanceNum())))

        imgH, imgW = self.bgrframe.shape[:2]
        resizedh, resizedw = self.resized_bgr.shape[:2]
        depth_h, depth_w = self.depthFrame.shape

        midW = int(depth_w / 2) # middle of the depth img width
        midH = int(depth_h / 2) # middle of the depth img height

        roi_df = deepcopy(data) 

        # rescale ROI (detected obj) from isp scale to match depthframe size for depth extraction
        roi_df["xmin2"] = (roi_df.xmin/imgW*depth_w).astype(int)
        roi_df["ymin2"] = (roi_df.ymin/imgH*depth_h).astype(int)
        roi_df["xmax2"] = (roi_df.xmax/imgW*depth_w).astype(int)
        roi_df["ymax2"] = (roi_df.ymax/imgH*depth_h).astype(int)

        # rescale ROI (detected obj) from isp scale to match frame_resized size to show detected obj
        roi_df["xmin"] = (roi_df.xmin/imgW*resizedw).astype(int)
        roi_df["ymin"] = (roi_df.ymin/imgH*resizedh).astype(int)
        roi_df["xmax"] = (roi_df.xmax/imgW*resizedw).astype(int)
        roi_df["ymax"] = (roi_df.ymax/imgH*resizedh).astype(int)

        roi_df["w"] = (roi_df.xmax2)-(roi_df.xmin2) 
        roi_df["h"] = (roi_df.ymax2)-(roi_df.ymin2) 

        """
        since we are not using calibrated lenspos when obtaining depthFrame, RGB and D may not align properly. to counter this we offset the roi inwards
        
        Conditions:
            1. Luxonis doc mentions that there should be a bout 10 X 10 px when obtaining depth
            2. for simplicity we scale the roi such that it is bboxScale * roi. if bboxScale == 0.5,
                - eg , x2 = x2_ori - 0.5(w_ori-w_new), 
                    if bboxScale * roi < 10 px, 
                        w_new = w_ori:  dont rescale
                    else:
                        w_new = bboxScale * roi 
        """
        roi_df["xmin2"] = roi_df.apply(lambda row : (row["xmin2"] + (row["w"]-clamp_px(row["w"]))/2), axis=1).astype(int)
        roi_df["xmax2"] = roi_df.apply(lambda row : (row["xmax2"] - (row["w"]-clamp_px(row["w"]))/2), axis=1).astype(int)
        roi_df["ymin2"] = roi_df.apply(lambda row : (row["ymin2"] + (row["h"]-clamp_px(row["h"]))/2), axis=1).astype(int)
        roi_df["ymax2"] = roi_df.apply(lambda row : (row["ymax2"] - (row["h"]-clamp_px(row["h"]))/2), axis=1).astype(int)

        # rescale depth ROI from depthframe scale -> frame_resized size to plot offsetROI
        roi_df["xmin3"] = (roi_df.xmin2/depth_w*resizedw).astype(int)
        roi_df["ymin3"] = (roi_df.ymin2/depth_h*resizedh).astype(int)
        roi_df["xmax3"] = (roi_df.xmax2/depth_w*resizedw).astype(int)
        roi_df["ymax3"] = (roi_df.ymax2/depth_h*resizedh).astype(int)

        # calc ROI centroid 
        roi_df["cx"] = ((roi_df["xmax2"] + roi_df["xmin2"]) / 2).astype(int) 
        roi_df["cy"] = ((roi_df["ymax2"] + roi_df["ymin2"]) / 2).astype(int) 

        roi_df["bb_xy_pos"] = roi_df.apply(lambda row : {"x": row["cx"] - midW,
                                                        "y": row["cy"] - midH}, axis=1)
        
        roi_df["angle_xy"] = roi_df.apply(lambda row : {"x":calc_angle(row["bb_xy_pos"]["x"],HFOV), 
                                                        "y":calc_angle(row["bb_xy_pos"]["y"],HFOV)}, axis=1)
        
        roi_df["averageDepth"] = roi_df.apply(lambda row : get_avgDepth(row["ymin2"],row["ymax2"],
                                                                        row["xmin2"],row["xmax2"],), axis=1) 

        roi_df["spatials"] = roi_df.apply(lambda row : {"z":row["averageDepth"],
                                                        "x":row["averageDepth"]* math.tan(row["angle_xy"]["x"]),
                                                        'y': -1*row["averageDepth"] * math.tan(row["angle_xy"]["y"])}, axis=1)

        
        """
        sample to dict
        {1: {'xmin': 198, 'ymin': 104, 'xmax': 547, 'ymax': 413, 'confidence': 0.9215731620788574, 
        'class': 21, 'name': 't0', 'xmin2': 241, 'ymin2': 142, 'xmax2': 503, 'ymax2': 374, 'w': 349, 'h': 309, 'xmin3': 241, 
        'ymin3': 142, 'xmax3': 503, 'ymax3': 373, 
        'cx': 372, 'cy': 258, 
        'bb_xy_pos': {'x': -28, 'y': -41}, 'angle_xy': {'x': -0.0478877981386183, 'y': -0.07006021554283412}, 
        'averageDepth': 617.0, 
        'spatials': {'z': 617.0, 'x': -29.5693781467733, 'y': 43.29801800063233}}}
        
        
        """
        
        return roi_df, roi_df.to_dict(orient="index")

    def get_bgrFrame_only(self):
        self.bgrData = self.rgbQueue.get()
        self.bgrframe = self.bgrData.getCvFrame()
        self.resized_bgr = po_utils.resize_img(self.bgrframe, width=950, inter = cv2.INTER_AREA)

    # frames
    def staticframe_get(self, focusMode : Optional[Dict[str, str]], target_lensPos : int = 0,
                        af_roi : Optional[po_utils.bbox_XY] = None, 
                        show_afroi : Tuple[bool, str] = (False, PO_CONST.CV_WIN_NAME.MAIN)):
        
        """
        example focuseMode:
            - None
            - {"mode" : "AFroi"}
        """
        
        def whileNotSync_getmsg():
            while not self.synced_msgs:
                self.camq_getmsg()
        
        MODE = "mode"
        
        try:
            if focusMode is None:
                self.camq_drain()
                whileNotSync_getmsg()
                self.staticframe_assign()

            elif focusMode[MODE] == "AFroi": 
                if af_roi is None:
                    af_roi = PredefinedAFRegion(*self.bgrframe.shape[:2]).default 
                    
                whileNotSync_getmsg()
                self.staticframe_assign()
                # self.waitAFcomplete()
                
                af_region_util = cam_utils.AFRegionUtil(AFROI = af_roi)
                af_roi_array = af_region_util.toRoi()
                
                self.setAFOnROI(af_roi_array)
                self.waitAFcomplete()
                
                self.camq_drain()
                whileNotSync_getmsg()
                self.staticframe_assign()
                
                if show_afroi[0]:
                    af_region_util.display_region(show_afroi[1], self.bgrframe)
            
            elif focusMode[MODE] == "lensPos":
                if target_lensPos == self.lensPos:
                    self.camq_drain()
                    whileNotSync_getmsg()
                        
                else:
                    lensPos = target_lensPos
                    self.setFocusManual(lensPos)
                    
                    whileNotSync_getmsg()
                    self.staticframe_assign()
                    self.waitAFcomplete()
                    
                    self.camq_drain()
                    while True:
                        self.camq_getmsg()
                        
                        if self.synced_msgs and self.synced_msgs[CAM_STREAM_NAMES['rgb']].getLensPosition() == lensPos:
                            break
                        
                self.staticframe_assign()
                self.gui_wins.update_valsNcolor(element_key='-autofocus_combo-', value = "PRESET")
            
            else:
                AF_mode = focusMode[MODE] # for focus mode that cames with oakd

                if AF_mode in PO_CONST.CAM_AF_OPT:
                    self.setFocusMode(AF_mode)
                    
                    if AF_mode != "OFF":
                        whileNotSync_getmsg()
                        self.staticframe_assign()
                        self.waitAFcomplete()
                    
                    self.camq_drain()
                    whileNotSync_getmsg()
                    self.staticframe_assign()
                        
                else:
                    self.camq_drain()
                    whileNotSync_getmsg()
                    self.staticframe_assign()
                    
                    message = f"Incorrect AF_mode : {AF_mode}. {PO_CONST.CBPPME}"
                    self.gui_wins.out_msg2term(message, lvl = "red", autolog=(True, log_level_keys.critical), popup = True)
            
            self.lensPos = self.synced_msgs[CAM_STREAM_NAMES['rgb']].getLensPosition()
            self.gui_wins.update_valsNcolor(element_key= "-lens_position-", value = self.lensPos)
            self.gui_wins.update_valsNcolor(element_key= "-AF_slider-", value = self.lensPos)
        
        except Exception:
            message = "Failed to get staticFrame."
            self.gui_wins.out_msg2term(message, lvl = "red", autolog=(True, log_level_keys.exception), popup=True)
            
    def staticframe_assign(self,  modes : Union[str, List[str]] = ["self"]) : # may or may not return
        """
        : modes 
            "ext" - return for ext use without assinging to instance attribute
            "self" - assign to attribute without returning
        """
        try:
            bgrframe = self.synced_msgs[CAM_STREAM_NAMES['rgb']].getCvFrame()  
            resized_bgr = cv2.resize(bgrframe, self.CAM_PARAM["stereo"]["outputsz"], interpolation=cv2.INTER_AREA) 
            
            static_depthmsg, depthFrame = None, None
            if GET_SYNC_FRAME:
                static_depthmsg = self.synced_msgs[CAM_STREAM_NAMES['depth']]  
                depthFrame : dai.ImgFrame = static_depthmsg.getFrame() 
                
            modes = po_utils.return_as_list(modes)

            for mode in modes:
                if mode == "ext": 
                    return bgrframe, resized_bgr, static_depthmsg, depthFrame
                elif mode == "self":
                    self.bgrframe = bgrframe
                    self.resized_bgr = resized_bgr
                    self.static_depthmsg = static_depthmsg
                    self.depthFrame = depthFrame

        except Exception as e:
            message = "cam_staticframe_assign failed"
            self.gui_wins.out_msg2term(message=message, lvl="red", 
                                       autolog=(True, "exception"), popup = True)

    def staticframe_assign_color(self, colorq_idx : np.uint8,  modes : Union[str, List[str]] = ["self"]):
        try:
            bgrframe = self.cam_outq[colorq_idx].get().getCvFrame()
            resized_bgr = cv2.resize(bgrframe, self.CAM_PARAM["stereo"]["outputsz"], 
                                     interpolation=cv2.INTER_AREA)

            modes = po_utils.return_as_list(modes)

            for mode in modes:
                if mode == "ext": 
                    return bgrframe, resized_bgr
                elif mode == "self":
                    self.bgrframe = bgrframe
                    self.resized_bgr = resized_bgr

        except Exception as e:
            message = f"cam_staticframe_assign_custom failed"
            self.gui_wins.out_msg2term(message=message, lvl="red", autolog=(True, "exception"), popup = True)
    
    
    # @ last update 08 jun 2023
    def cam_dispSynced(self, get_ext_frame : Tuple[bool, po_utils.Frame] = \
                       (False, po_utils.Frame())):
        
        cv_win_name = PO_CONST.CV_WIN_NAME.MAIN
        # fps.nextIter()
        # print('FPS', fps.fps())

        if not get_ext_frame[0]:
            frame = self.resized_bgr.copy() 
            depth = self.depthFrame[self.mouse_point[1]][self.mouse_point[0]] 
        else:
            frame = get_ext_frame[1].resized_bgr.copy()
            depth = get_ext_frame[1].depthFrame[self.mouse_point[1]][self.mouse_point[0]] 
        
        THelper.put_depth(frame, self.mouse_point, depth)

        po_utils.cv2show(cv_win_name, frame)

    # AF
    def setAFOnROI(self, roi: NDArray[np.int_]):
        """
        roi : [pos x , pos y, size x , size y]
        """
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusRegion(*roi)
        
        try:
            self.controlQueue.send(ctrl)
            self.gui_wins.update_valsNcolor(element_key='-autofocus_combo-', value= "ROI")
        except AttributeError:
            pass 
    
    # @ last update 06 jun 2023
    def setFocusManual(self, lensPos : int, set_af_combo : bool = True, 
                       set_slider_val : bool = True): 
        # ! Warning. After setting manual focus, it impossible to set AutoFocusRegion without first setting any of dai.CameraControl.AutoFocusMode (excluding "OFF"). 
        # ! best use AUTO or CONT_VID after manualFocus
        self.lensPos = lensPos
        
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(lensPos)

        try:
            self.controlQueue.send(ctrl)

            if set_af_combo:
                self.gui_wins.update_valsNcolor(element_key='-autofocus_combo-', value = "USER_DEFINE")
            
            self.gui_wins.update_valsNcolor(element_key="-lens_position-", value = lensPos)

            if set_slider_val:
                self.gui_wins.update_valsNcolor(element_key="-AF_slider-", value = lensPos)

        except AttributeError:
            pass 

    def setFocusMode(self, focusMode : str): # @ last update 06 jun 2023
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(CAM_AF_OPT[focusMode]) 
        ctrl.setAutoFocusTrigger()

        try:
            self.controlQueue.send(ctrl)
            self.gui_wins.f_AF(focusMode)
        except AttributeError:
            pass 
    
    # @ update 09 jun 2023
    def setFocus_w_GUI(self) -> None:
        if self.gui_wins.event == '-autofocus_combo-': 
            AF_mode = self.gui_wins.values['-autofocus_combo-']

            if AF_mode in CAM_AF_OPT:
                self.setFocusMode(AF_mode)

                if AF_mode != "OFF":
                    self.waitAFcomplete()
                
                if GET_SYNC_FRAME:
                    self.camq_drain()

                    while not self.synced_msgs:
                        self.camq_getmsg()

                    self.staticframe_assign()

                    lensPos = self.synced_msgs[CAM_STREAM_NAMES['rgb']].getLensPosition()

                else:
                    # for dataset collector mode
                    self.get_bgrFrame_only()
                    lensPos = self.bgrData.getLensPosition()

                self.lensPos = lensPos

                self.gui_wins.update_valsNcolor(element_key="-lens_position-", value = lensPos)
                self.gui_wins.update_valsNcolor(element_key="-AF_slider-", value = lensPos)

            elif AF_mode == "AFroi":
                bb_drawer = ManualBBoxDrawer(self, self.gui_wins)
                bb_drawer.draw_rectangle()

                af_region_util = cam_utils.AFRegionUtil(AFROI = bb_drawer.bbox_wrt_bgrframe)
                af_roi_array = af_region_util.toRoi()

                self.setAFOnROI(af_roi_array)
                self.waitAFcomplete()

                af_region_util.display_region(bb_drawer.window_name,
                                               self.bgrframe)

        elif self.gui_wins.event in [',', '.',]:
            lensPos = self.lensPos

            if self.gui_wins.event == ',':
                lensPos -= 1
            if self.gui_wins.event == '.':
                lensPos += 1

            lensPos = po_utils.clamp(lensPos, 0, 255)
            self.setFocusManual(lensPos)

        elif self.gui_wins.event in ["-AF_slider-"]:
            lensPos = int(self.gui_wins.values[self.gui_wins.event])
            self.setFocusManual(lensPos, set_slider_val = False)

    # queues
    def camq_drain(self): # @ update 09 jun 2023
        self.synced_msgs = False
        self.sync = HostSync_ts() # empty container before starting to get frames from cam again

        # empty old images from oak outputqueue before putting in new frames into container
        for q in self.cam_outq:
            while q.has():
                # sometimes after tryGetAll(), q.has() == True. using while, sometimes we need to tryGetAll() twice  
                q.tryGetAll()

    def camq_getmsg(self): # @ update 07 jun 2023
        for q in self.cam_outq:
            if q.has():
                self.synced_msgs = self.sync.add_msg(q.getName(), q.get())

    def set_dispaQ(self): # @ last update 06 jun 2023
        dispaQ = None
        
        if GET_DISPA_FRAME:
            cv2.namedWindow(PO_CONST.CV_WIN_NAME.DISPA)
            dispaQ = self.device.getOutputQueue(self.CAM_PARAM["stream_names"]["disparity"], 1, False)
        
        return dispaQ
    
    def set_qSysInfo(self): # @ last update 06 jun 2023
        qSysInfo = None

        if self.SES_CAM_SETTING["get_oak_log"]:
            qSysInfo = self.device.getOutputQueue(name=self.CAM_PARAM["stream_names"]["oak_log"], 
                                                  maxSize=4, blocking=False) 
            
        return qSysInfo

    # laser
    def setIRLaser(self):
        laser_mA = self.gui_wins.values[self.gui_wins.event]
        self.device.setIrLaserDotProjectorBrightness(laser_mA)

    def send_ctrl_to_cam(self, ctrl : dai.CameraControl):
        try:
            self.controlQueue.send(ctrl)
        except AttributeError:
            pass 

    def AutoExposure_enable(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.send_ctrl_to_cam(ctrl)

    def AutoWhiteBalance_enable(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        self.send_ctrl_to_cam(ctrl)

    def Exposure_set_manual(self):
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self.ctrl_param.expTime, self.ctrl_param.sensIso)
        self.send_ctrl_to_cam(ctrl)
    
    def WhiteBalance_set_manual(self):
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(self.ctrl_param.wbManual)
        self.send_ctrl_to_cam(ctrl)

    def AWBalance_lock_set(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceLock(self.ctrl_param.awb_lock)
        self.send_ctrl_to_cam(ctrl)

    def AutoExposure_lock_set(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureLock(self.ctrl_param.ae_lock)
        self.send_ctrl_to_cam(ctrl)

    # others
    def set_nonAF_settings(self) -> None: # @ update 09 jun 2023
        if self.gui_wins.event == "-laser_slider-":
            self.setIRLaser()

    def waitAFcomplete(self, t : Optional[Union[int, float]] = None, 
                       ): # @ last update 06 jun 2023
        # for i in tqdm(range(3), desc="AF Countdown", unit="s", ncols=80, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]'):
        #     time.sleep(1) 

        if t is None:
            t = self.wait_AF_duration

        txt = f"AF in progress...wait {t} seconds"
        THelper.putText_mid(self.resized_bgr, txt, font_color=(0,149,255))

        if COBOT_ENABLED:
            po_utils.cv2show(PO_CONST.CV_WIN_NAME.MAIN, self.resized_bgr)
        else:
            imgbytes=cv2.imencode('.png', self.resized_bgr)[1].tobytes()
            self.gui_wins.update_valsNcolor(element_key = "-GRAPH-", data=imgbytes)
            self.gui_wins.main_win.refresh()

        time.sleep(t)

    def waitSteady(self, t : Union[int, float] = 0.2): # @ last update 06 jun 2023
        time.sleep(t)

    def debugger_cvimshow(self, img: NDArray[np.int_], title : str = PO_CONST.CV_WIN_NAME.DEBUG, 
                      auto_resize : bool = True): # @ last update 06 jun 2023
        cv2.namedWindow(title)

        if img.shape[0]>800 and auto_resize: 
            cv2.imshow(title, po_utils.resize_img(img, width=800)) 
        else:
            cv2.imshow(title, img)

        while True:
            self.gui_wins.read_all()

            if po_utils.cv2_custom_quit(title):
                break

    def show_distance_at_mousepoint(self, event, x, y, args, params):
        self.mouse_point = (x, y)

    def getColorQ_index(self): # @ last update 06 jun 2023
        color_qname = self.CAM_PARAM["stream_names"]["rgb"]
        colorq_idx = -1

        qnames = [q.getName() for q in self.cam_outq]
    
        if color_qname in qnames:
            colorq_idx = qnames.index(color_qname)
        else: 
            message = f"colorq_idx: {color_qname} does not exist in queue."
            logger.critical(message)
            gui.popup(message, title="Warning")
        
        return colorq_idx
    
    def rgb_only_whileLoop(self,): # @ last update 06 jun 2023
        cv2_windowname = PO_CONST.CV_WIN_NAME.COLOR

        colorq_idx = self.getColorQ_index() 
        if colorq_idx>=0:
            try:
                while True:
                    self.gui_wins.read_all()
                    
                    bgrframe = self.cam_outq[colorq_idx].get().getCvFrame()
                    resized_bgr = cv2.resize(bgrframe, 
                                             self.CAM_PARAM["stereo"]["outputsz"], interpolation=cv2.INTER_AREA)  

                    cv2.imshow(cv2_windowname, resized_bgr) 

                    # key wait needs to be after cvshow
                    if po_utils.cv2_custom_quit(cv2_windowname):
                        break

            except Exception as e:
                message = f"Failed cam_rgb_only_whileLoop. {PO_CONST.CBPPME}"
                self.gui_wins.out_msg2term(message, lvl="red", autolog=(True, log_level_keys.exception), popup = True)

    def reset_main_win_mousecallback(self,):
        cv2.setMouseCallback(PO_CONST.CV_WIN_NAME.MAIN, self.show_distance_at_mousepoint)

    def __custom_setter(self, gui_wins : GUIWindows): # @ last update 06 jun 2023
        # not necessary to use this method but putting it here instead of __init__ allows vs show the methods
        self.gui_wins : GUIWindows = gui_wins 
        self.sync = HostSync_ts()

@define(slots=True)
class ManualBBoxDrawer:
    """
    A tool to draw a single bbox on image
    """
    cam : Camera
    gui_wins : GUIWindows
    
    is_drawing : bool = False
    # window_name : str = "ManualBBoxDrawer"
    window_name : str = PO_CONST.CV_WIN_NAME.MAIN

    bgrframe : Optional[NDArray[np.uint8]] = None
    resized_bgr : Optional[NDArray[np.uint8]] = None
    resized_bgr_with_bb : Optional[NDArray[np.uint8]] = None

    # cooridnates of bbox
    rect_start : Optional[Tuple[int, int]] = None
    rect_end : Optional[Tuple[int, int]] = None
    bbox_wrt_resized_bgr : po_utils.bbox_XY = field(factory=po_utils.bbox_XY)
    bbox_wrt_bgrframe : po_utils.bbox_XY = field(factory=po_utils.bbox_XY)

    def draw_rectangle(self): 
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        proceed = False

        if GET_SYNC_FRAME:
            colorq_idx = self.cam.getColorQ_index()
            if colorq_idx>=0:
                self.cam.staticframe_assign_color(colorq_idx)
                proceed = True
        else:
            self.cam.get_bgrFrame_only()
            proceed = True

        if proceed :
            rsz_h, rsz_w = self.cam.resized_bgr.shape[:2]
            h, w = self.cam.bgrframe.shape[:2]

            while True:
                self.gui_wins.read_all()

                if GET_SYNC_FRAME:
                    self.cam.staticframe_assign_color(colorq_idx)
                else:
                    self.cam.get_bgrFrame_only()

                self.resized_bgr_with_bb = self.cam.resized_bgr.copy()

                if self.rect_start and self.rect_end:
                    cv2.rectangle(self.resized_bgr_with_bb, self.rect_start, 
                                    self.rect_end, (0, 255, 0), 2)

                if self.rect_start:
                    cv2.putText(self.resized_bgr_with_bb, 
                                f"Box start: {self.rect_start}",
                                (10, 30), PO_CONST.FONTFACE, 1, (0, 255, 0), 2)

                if self.rect_end:
                    cv2.putText(self.resized_bgr_with_bb, f"Box end: {self.rect_end}",
                                    (10, 70), PO_CONST.FONTFACE, 1, (0, 255, 0), 2)

                cv2.imshow(self.window_name, self.resized_bgr_with_bb)
                key = cv2.waitKey(1)
                if key == ord('r'):  # Reset rectangle
                    self.rect_start = None
                    self.rect_end = None
                elif key == ord('q') or key == 27 or self.gui_wins.event == "-capture-":
                    # Quit on 'q' or ESC
                    break 

            self.reorder_coordinates(rsz_w, rsz_h, w , h)

            self.resized_bgr = self.cam.resized_bgr
            self.bgrframe = self.cam.bgrframe

            if self.window_name == PO_CONST.CV_WIN_NAME.MAIN:
                self.cam.reset_main_win_mousecallback()

    def reorder_coordinates(self, rsz_w, rsz_h, w , h):
        # user may draw from a coordinates of bottom right to top left which will cause issue to other codes. this prevents issues.
        # coordintaes clamped to stay within img frame dimensions (no -ve, cnnt exceed frame max dimension)
        ordered_start : Tuple[int, int] = (po_utils.clamp(min(self.rect_start[0],self.rect_end[0]), 0, rsz_w),
                                            po_utils.clamp(min(self.rect_start[1],self.rect_end[1]), 0, rsz_h))
        
        ordered_end : Tuple[int, int] = ( po_utils.clamp(max(self.rect_start[0],self.rect_end[0]), 0, rsz_w),
                                        po_utils.clamp(max(self.rect_start[1],self.rect_end[1]), 0, rsz_h))

        self.bbox_wrt_resized_bgr = po_utils.bbox_XY(ordered_start[0], ordered_start[1], 
                                                     ordered_end[0], ordered_end[1])
        
        self.bbox_wrt_bgrframe = po_utils.bbox_XY(int(ordered_start[0]/rsz_w*w), int(ordered_start[1]/rsz_h*h), 
                                             int(ordered_end[0]/rsz_w*w), int(ordered_end[1]/rsz_h*h))
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_drawing:
                self.rect_start = None
                self.rect_end = None
                self.is_drawing = False
            else:
                self.rect_start = (x, y)
                self.rect_end = None
                self.is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if self.is_drawing:
                self.rect_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False 

    """
    A tool to draw a single bbox on image
    """
    cam : Camera
    gui_wins : GUIWindows
    
    is_drawing : bool = False
    # window_name : str = "ManualBBoxDrawer"
    window_name : str = PO_CONST.CV_WIN_NAME.MAIN

    bgrframe : Optional[NDArray[np.uint8]] = None
    resized_bgr : Optional[NDArray[np.uint8]] = None
    resized_bgr_with_bb : Optional[NDArray[np.uint8]] = None

    # cooridnates of bbox
    rect_start : Optional[Tuple[int, int]] = None
    rect_end : Optional[Tuple[int, int]] = None
    bbox_wrt_resized_bgr : po_utils.bbox_XY = field(factory=po_utils.bbox_XY)
    bbox_wrt_bgrframe : po_utils.bbox_XY = field(factory=po_utils.bbox_XY)

    def draw_rectangle(self): 
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        proceed = False

        if GET_SYNC_FRAME:
            colorq_idx = self.cam.getColorQ_index()
            if colorq_idx>=0:
                self.cam.staticframe_assign_color(colorq_idx)
                proceed = True
        else:
            self.cam.get_bgrFrame_only()
            proceed = True

        if proceed :
            rsz_h, rsz_w = self.cam.resized_bgr.shape[:2]
            h, w = self.cam.bgrframe.shape[:2]

            while True:
                self.gui_wins.read_all()

                if GET_SYNC_FRAME:
                    self.cam.staticframe_assign_color(colorq_idx)
                else:
                    self.cam.get_bgrFrame_only()

                self.resized_bgr_with_bb = self.cam.resized_bgr.copy()

                if self.rect_start and self.rect_end:
                    cv2.rectangle(self.resized_bgr_with_bb, self.rect_start, 
                                    self.rect_end, (0, 255, 0), 2)

                if self.rect_start:
                    cv2.putText(self.resized_bgr_with_bb, 
                                f"Box start: {self.rect_start}",
                                (10, 30), PO_CONST.FONTFACE, 1, (0, 255, 0), 2)

                if self.rect_end:
                    cv2.putText(self.resized_bgr_with_bb, f"Box end: {self.rect_end}",
                                    (10, 70), PO_CONST.FONTFACE, 1, (0, 255, 0), 2)

                cv2.imshow(self.window_name, self.resized_bgr_with_bb)
                key = cv2.waitKey(1)
                if key == ord('r'):  # Reset rectangle
                    self.rect_start = None
                    self.rect_end = None
                elif key == ord('q') or key == 27 or self.gui_wins.event == "-capture-":
                    # Quit on 'q' or ESC
                    break 

            self.reorder_coordinates(rsz_w, rsz_h, w , h)

            self.resized_bgr = self.cam.resized_bgr
            self.bgrframe = self.cam.bgrframe

            if self.window_name == PO_CONST.CV_WIN_NAME.MAIN:
                self.cam.reset_main_win_mousecallback()
                
    def reorder_coordinates(self, rsz_w, rsz_h, w , h):
        # user may draw from a coordinates of bottom right to top left which will cause issue to other codes. this prevents issues.
        # coordintaes clamped to stay within img frame dimensions (no -ve, cnnt exceed frame max dimension)
        ordered_start : Tuple[int, int] = (po_utils.clamp(min(self.rect_start[0],self.rect_end[0]), 0, rsz_w),
                                            po_utils.clamp(min(self.rect_start[1],self.rect_end[1]), 0, rsz_h))
        
        ordered_end : Tuple[int, int] = ( po_utils.clamp(max(self.rect_start[0],self.rect_end[0]), 0, rsz_w),
                                        po_utils.clamp(max(self.rect_start[1],self.rect_end[1]), 0, rsz_h))

        self.bbox_wrt_resized_bgr = po_utils.bbox_XY(ordered_start[0], ordered_start[1], 
                                                     ordered_end[0], ordered_end[1])
        
        self.bbox_wrt_bgrframe = po_utils.bbox_XY(int(ordered_start[0]/rsz_w*w), int(ordered_start[1]/rsz_h*h), 
                                             int(ordered_end[0]/rsz_w*w), int(ordered_end[1]/rsz_h*h))
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_drawing:
                self.rect_start = None
                self.rect_end = None
                self.is_drawing = False
            else:
                self.rect_start = (x, y)
                self.rect_end = None
                self.is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if self.is_drawing:
                self.rect_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False 