from typing import Dict, Tuple, Union, List, Optional, Pattern, Final, Any
from attrs import asdict, define, make_class, Factory, field
import numpy as np
import depthai as dai
import datetime
from itertools import cycle
import PySimpleGUI as gui
import pathlib
import cv2
import pycountry
import re 
import pathlib

"""
t0
"""

# product faces
PROD_FACES = ("top", "front", "back", "left", "right")

# common strs
@define(slots=True, frozen=True)
class ConstantStrs():
    CALDONE : str = "CalDone" 
    COC : str = "CertificateofCalibration"
    COO : str = "COO"
    DATA : str = "data"
    DATEOCALIB :str = "DateofCalibration"
    DATECALDUE : str = "CalDueDate"
    DATEP : str = "PrintDate"
    DATEU : str = "DateUsed"
    DIST_CAM2SUB : str = "dist_cam2sub"
    DOCA4 : str = "docA4"
    IN : str = "ItemNo"
    JN  : str = "JobNo"
    LENSPOS : str = "lensPos"
    MD: str = "missing_data"
    MISSING : str = "missing"
    MN : str = "ModelNo"
    OK : str = "ok"
    OPTS : str = "Options"
    PASSED : str = "passed"
    PG : str = "Page"
    SN : str = "SerialNo"
    SN2 : str = "S/N"
    
CONSTANT_STR = ConstantStrs()
DIST_CAM2SUB = CONSTANT_STR.DIST_CAM2SUB
LENSPOS = CONSTANT_STR.LENSPOS

# shared
OPT_LEN : int = 3
BIGFOOT_OPT = {'81133A':"",'81134A':"",'M8020A':"",'M8040A':"",
               'M8041A':"0A20A30G20G30G40G50G60G70S10S20S30S40S60SXC08C16DEMG08G16U14U16UA2UA3UEDUG2UG3UG4UG5UG6UG7US1US2US3US4US6USX",
               'M8045A':"0G20G30G40G60G90P30P6DEMG32G64U64UG2UG3UG4UG6UG9UP3UP6",
               'M8046A':"0A30A40A50N20P30P60S10S20S30S40S6A32A64DEMU64UA3UA4UA5UN2UP3UP6US1US2US3US4US6",'M8047A':"",
               'M8048A':"001002",'M8049A':"001002003",
               'M8050A':"",'M8051A':"0A20A30G20G30G40G50G70S60SXC08C16G08G16U16UA2UA3UEDUG2UG3UG4UG5UG7US6USX",
               'M8054A':"004",'M8057A':"",'M8057B':"",'M8190A':"00100202G12G14BALLAMPB02DUCFSWLPNSEQ",'M8190S':"",
               'M8192A':"",'M8194A':"001002004",'M8195A':"00100200416GE14FSWR12R14SEQ",'M8196A':"001002004",
               'M8197A':"",'M8199A':"002004",'M9502A':"",'M9505A':"",'N4951B':"",'N4952A':"",'N4960A':"",'N4962A':""}
NON_EXISTENT_INT = -2

CLOSEUP_SUBJECTS : List[str] = ["Lkcc", "t0", "Lptd", "Lst"]
CLOSEUP_LABELS : List[str] = [x for x in CLOSEUP_SUBJECTS if x not in ["t0"]]

@define(slots=True, frozen=True)
class CheckerKey:
    mn_sn_opt_checker : str = f"{CONSTANT_STR.MN}_SN_OPT result"
    sn_seeker : str = f"{CONSTANT_STR.SN}_matcher result"
    

@define(slots=True, frozen=True)
class FileStorageConsts:
    coc : str = "coc"
    coc_that_passed : str = "coc_"
    
    label_closeup_result : str = "Lclosup_result.txt"
    
    smcc : str =  "submodel_results"
    
    labelndoc_insp_res_sum : str = "label_and_doc_insp_results_summary.txt"

@define(slots=True, frozen=True)
class ImgStorageConsts:
    Lcloseup_subfolder : pathlib.Path = pathlib.Path("label_closeup")
    
    masked_file_name : str = "_masked" # img that have been masked
    detL_file_name : str = "_detL" # img of labels with bbox plotted

FILE_STORAGE_CONST = FileStorageConsts()
IMG_STORAGE_CONST = ImgStorageConsts()

# dates
MONTHS = "January February March April May June July August September October November December"
# MONTHS_LIST = [datetime.date(2022, m, 1).strftime('%m - %B') for m in range(1, 13)]
MONTHS_LIST = [datetime.date(2022, m, 1).strftime('%B').upper() for m in range(1, 13)]

months_1 = [x[0].upper() for x in MONTHS.split()]
months_2 = [x[1].upper() for x in MONTHS.split()]

# doc
@define(slots=True, frozen=True)
class DocsRequiringInspection:
    coc : str = "coc"
    coe : str = "coe"
    M8070B_doc : str = "M8070B_doc"

@define(slots=True, frozen=True)
class DocNames:
    COC : str = field(default=CONSTANT_STR.COC)
    
# coc
CERTIFICATE_OF_CALIBRATION = CONSTANT_STR.COC
DATE_USED = CONSTANT_STR.DATEU
CAL_DUE_DATE = CONSTANT_STR.DATECALDUE
DATE_OF_CALIBRATION = CONSTANT_STR.DATEOCALIB
PRINT_DATE = CONSTANT_STR.DATEP

"""
example usage of regex
[<regex>.findall(data) for data in <List[str]>]
"""
Pages_regex = re.compile(r"[0-9]+") # regex for getting numbers from a str
regex_1 = re.compile("(?<=[:]).*") # regex for finding txt after ":". 
regex_2 = re.compile("(?<=[#]).*") # regex for finding txt after "#"
date_regex =  re.compile(rf"[0-3][0-9][-]{months_1}{months_2}.[-]....", re.IGNORECASE)

coc_var : Dict[str, Dict[str, str]] = { 
        CONSTANT_STR.PG : {"var": CONSTANT_STR.PG},
        CONSTANT_STR.MN : {"var" : CONSTANT_STR.MN,},
        CONSTANT_STR.SN : {"var" : CONSTANT_STR.SN,},
        # "Options":{"var":"Options","regex":regex_1},
        CONSTANT_STR.OPTS : {"var": f"{CONSTANT_STR.OPTS}InstalledWithSpecifications"},
        CONSTANT_STR.DATEOCALIB : {"var": CONSTANT_STR.DATEOCALIB,},
        CONSTANT_STR.DATEP : {"var": CONSTANT_STR.DATEP,},
        CONSTANT_STR.SN2 :{"var" : CONSTANT_STR.SN2},
        CONSTANT_STR.CALDONE : {"var" : CONSTANT_STR.CALDONE}
        }

coc_tol_pix : int = 30
coc_date_format : str = '%d-%b-%Y' # used to extract date_used and , cal_due_date

# paths
NAS_PATH = pathlib.Path(r"\\pngnas1.png.is.keysight.com\bppvdrive")
SES_SETTING_YAML = r"./polaris_pkg/SESSION_SETTINGS.yaml"
LABEL_AND_DOC_CHECKLIST_YAML = r"./polaris_pkg/configfile.yaml" # inspection config

# cobot
# @define(slots=True, frozen=True)
# class CobotCommandStings():
#     method : str = "method"
#     movel_tcp : str = "movel_tcp"
#     p_to : str = "p_to"
    
# COBOT_STR =  CobotCommandStings()

COBOT : Dict = {"sim" : # UR sim cobot
            {"av": (0.4, 0.4),
            "ROBOT_IP": '192.168.169.128'}, 
        "phy" :  # physical cobot
            {"av" : (0.3, 0.3),
            "ROBOT_IP" : "10.66.108.83" }}

# distance [mm] the cobot will be from subject the lenspos for rgb focus
# ! Note that dist_cam2sub are += CAM2SUB__DIST_TOL
CAM2SUB_DIST_TOL = 35 # [mm]
COBOT_CAM_2SUBJECT= {"t0":{DIST_CAM2SUB:320, LENSPOS:135}, #10 apr 2023
                    "docA4":{DIST_CAM2SUB:305,LENSPOS:139},
                    "Lptd": {DIST_CAM2SUB:115, LENSPOS:167}, #10 apr 2023
                    "Lst":{DIST_CAM2SUB:115, LENSPOS:167}, 
                    # Lst 10 apr 2023 DIST_CAM2SUB:80, LENSPOS:255 ; #03 may 2023 testing with "Lptd" values instead of that on 10 apr 2023
                    "Lkcc":{DIST_CAM2SUB:115, LENSPOS:169}, #10 apr 2023
                    "tpod" : {DIST_CAM2SUB:182}
                    }
COBOT_CAM_2SUBJECT_FRONT = {"prod_0_front":{DIST_CAM2SUB:280}, # module
                            "frame_0_front":{DIST_CAM2SUB:380}, # chassis
                            "prod_1_front":{DIST_CAM2SUB:190}, # pod eg M8057
                            }
DISTANCE = {"cobotbase2cart":{"front2front": 720}, #v0: frnt2frnt 530
            "cam2cartfront":{"tol":20},
            "cobotSetup":{"urbase2basefront":120,
                          "tcp2camfront": 30.931,
                          "tcp2camcentre_vert": 70.264}} # get fom cad
for val in COBOT_CAM_2SUBJECT.values():
    val[DIST_CAM2SUB] += CAM2SUB_DIST_TOL
for val in COBOT_CAM_2SUBJECT_FRONT.values():
    val[DIST_CAM2SUB] += CAM2SUB_DIST_TOL

# ocr
IMG_MAX_DIMENSION = int(4056)
# a new addition to take ocr random punctuation mark
my_punct = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
             ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', 
           '`', '{', '|', '}', '~', '»', '«', '“', '”']
punct_regex = re.compile("[" + re.escape("".join(my_punct)) + "]")

# labels
cd_pix_tol : Final = 15 # tolerance required to crop slightly larger than boundingbox returned from obj det, otherwise may chip into datamatrix, leads to decoding failure
labels_requiring_ocr : Final = ["Lccr","Lfw","Lfda", "Lsafety1M","LfwM8070B","LfwM8070A/B","LfwM9505A"]

@define(slots=True, frozen=True)
class CV2WinNames():
    MAIN: str = "cam_output"
    DISPA: str = "disparity"
    COLOR: str = "color_only"
    DEBUG: str = "debugger"

@define(slots=True, frozen=True)
class LabelObjDetSetting():
    pt_file : str
    yaml_file : str
    infer_sz: int
    conf : int

CAM_PARAM = {"rgb":{"fps":30},
            "monoL":{"fps":30},
            "monoR":{"fps":30},
            "stereo":{"thresholdFilter": {"high":1000,"low":150},
                    "outputsz": (800, 599)},
            "stream_names":{"rgb":"rgb",
                            'ctrl':"control",
                            "disparity" : "disp",
                            "depth" : "depth",
                            "isp" : "isp",
                            "oak_log": "sysinfo"}}
CAM_CTR_PARAM = {LENSPOS : 150,
                "expTime" : 20000,
                "sensIso" : 800 ,   
                "wbManual" : 4000,
                "ae_comp" : 0,
                "ae_lock" : False,
                "awb_lock" : False,
                "saturation" : 0,
                "contrast" : 0,
                "brightness" : 0,
                "sharpness" : 0,
                "luma_denoise" : 0,
                "chroma_denoise" : 0,  
                "STEP_SIZE" : 8 ,# Step size ('W','A','S','D' controls)
                "EXP_STEP" : 500 , # us# Manual exposure/focus/white-balance set step
                "ISO_STEP" : 50,
                "LENS_STEP" : 3,
                "WB_STEP" : 200,
                "awb_mode":  cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()]),
                "anti_banding_mode": cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()]),
                "effect_mode": cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])}

@define(slots=True)
class CamControlParam:
    control : Any = 'none'
    lensPos : int =  150
    expTime : int = 20000
    sensIso : int = 800    
    wbManual : int = 4000
    ae_comp : int = 0
    ae_lock : bool =  False
    awb_lock : bool =  False
    saturation : int = 0
    contrast : int = 0
    brightness : int = 0
    sharpness : int = 0
    luma_denoise : int = 0
    chroma_denoise : int = 0  
    STEP_SIZE : int = 8 # Step size ('W','A','S','D' controls)
    EXP_STEP : int = 500  # us# Manual exposure/focus/white-balance set step
    ISO_STEP : int = 50
    LENS_STEP : int = 3
    WB_STEP : int = 200
    awb_mode : cycle = field(factory = lambda: cycle([item for name,
                               item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()]))
    anti_banding_mode : cycle  = field(factory = lambda: cycle([item for name,
                                        item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()]))
    effect_mode : cycle = field(factory = lambda: cycle([item for name,
                                  item in vars(dai.CameraControl.EffectMode).items() if name.isupper()]))

CAM_AF_OPT = {'AUTO_ONCE' :dai.CameraControl.AutoFocusMode.AUTO, 
              'CONT_VID': dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO,
              'MACRO': dai.CameraControl.AutoFocusMode.MACRO,
              "CONT_PIC":dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE,
              "EDOF":dai.CameraControl.AutoFocusMode.EDOF,
              "OFF": dai.CameraControl.AutoFocusMode.OFF}

CLR_CAM_RESO = (3040, 4056) #h, w
FRAME_DISP_SZ = (800, 599) # w,h


CBPPME = "Please contact BP PME."
PLS = "Please "
ASK_USER_MANUALLY_CHECK_COO = f"{PLS} manually check Country of Origin."
PERFORM_MANUAL_I = f"perform manual inspection."
PRESET_MSG = {"coc_pass" :  {
                "message": "COC: passed", "lvl": "g", "autolog" : (True, "info")},
              "coc_data_extracted" : {
                  "message": "coc_data_extracted", "autolog" : (True, "info")},
            "doc_pressed" : {
                "message" : "Doc pressed.",
                "lvl":"purple", "auto_newline":False, "autolog" : (True, "info")},
            "no_doc_found" : {
                "message" : "No document found",
                "lvl" : "red", "autolog" : (True, "info"), "popup" : True},
            "doc_predefined_title_not_found": {
                "message" : "Predefined title not found in document ocr, please perform manual inspection.", 
                 "lvl":"red", "autolog": (True, "info"), "popup" : True},
            "product/label_pressed" : {
                "message" : "Product/Label pressed.",
                "lvl":"purple","auto_newline":False, "autolog" : (True, "info")},
            "cobot_frontpos_warning" : {
                "message":"Ensure product & cart are at the right distance from cobot base otherwise COLLISION may occur.",
                "lvl":"o", 
                "autolog": (True, "info"),
                "popup": (True, "Warning")},
            "failed_to_decode_datamatrix":{
                "message" : "Failed to extract cd from image.Either\n1. the image is not clear (not in focus/camera too far) or\n2. datamatrix print quality is poor.",
                "lvl" : "red",
                "autolog": (True, "exception"),
                "popup": (True, "Warning")},
            "i_or_m_county":{
                "message" : "Incorrect or missing Country of Origin",
                "lvl" : "red",
                "autolog": (True, "info"),
                "popup": (True, "Warning")},
            "no_label_detected": {
                "message":"Algorithm did not detect any labels",
                "t_color":'white', "bg_color":'red',"exp_msg":"RPT", 
                "autolog" : (True, "info"), 
                "popup" : False},
            "img_analysis_comp": {
                "message" : "Image analysis complete",
                "t_color":'white', "bg_color":'green',"exp_msg":"RPT", "autolog" : (True, "info")},
            "cross_checker_triggered":{
                "message" : "Cross checker triggered.",
                "lvl":"purple"
            },
            "coc_vs_ptd_vs_st": {
                "message":"CoC vs PTD vs Serial tag pressed.",
                "lvl":"purple", "auto_newline":False, "autolog" : (True, "info")},
            "check_MN_SN_OPT":{
                "message" : "Checking bundle ModelNo, SerialNo, Options.",
                "lvl" : "purple",
            },
            "not_implemented":{
                "message" : f"!!! NOT IMPLEMENTED !!!\nPlease contact BP PME, and {PERFORM_MANUAL_I}",
                "lvl" : "red",
                "popup" : True,
                "autolog" : (True, "critical"), 
            },
            "preset_msg_error":{
                "message" : "Preset message not found",
                "t_color":'white', "bg_color":'red', "exp_msg":"RPT", "autolog" : (True, "debug")}, 

            }


# countries
COUNTRY_LIST = list(pycountry.countries)
COUNTRIES = [countries.name.capitalize() for countries in COUNTRY_LIST]
COUNTRY_ALPHA2_CODE = [x.alpha_2 for x in COUNTRY_LIST]

for index,text in enumerate(COUNTRIES):
    if text.startswith("Taiwan"):
        COUNTRIES[index] = "Taiwan".capitalize()

# visually distinct colors for plotting bounding boxes
HEX_CLR = ["#696969","#556b2f","#8b4513","#228b22","#483d8b","#008b8b","#4682b4","#9acd32","#00008b","#8fbc8f","#8b008b",
"#b03060","#ff0000","#ff8c00","#ffd700","#7fff00","#8a2be2","#00ff7f","#dc143c", "#00ffff","#f4a460","#0000ff",
"#f08080","#ff00ff","#1e90ff","#f0e68c","#dda0dd","#90ee90","#ff1493","#ffefd5"]

HEX2RGB : List[Tuple[int,int,int]] = []
for hex in HEX_CLR:
    hex = hex.lstrip('#')
    HEX2RGB.append(tuple(int(hex[i:i+2], 16) for i in (0, 2, 4)))

FONTSCALE = 1.5
FONTTHICK = 6
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX

#### instances of attrs ####
CV_WIN_NAME = CV2WinNames()


