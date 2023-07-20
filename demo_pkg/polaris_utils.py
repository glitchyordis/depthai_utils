from typing import Any, List, Union, Dict, Tuple, Optional
from attrs import asdict, define, field
import pandas as pd
import math
import time
import datetime
import matplotlib.pyplot as plt
# import inspect
import cv2
from numpy import typing as npt
from . import  POLARIS_CONSTANTS as PO_CONST
import numpy as np
import PySimpleGUI as gui
import traceback
import sys
from IPython.display import display_html
from pprint import pprint
from collections import Counter
import re
import torch
from torchvision.ops import box_iou
from .polaris_logger import logger

# This module contains helper functions for the main program


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255) 
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

        self.FONTSCALE = 1.5
        self.FONTTHICK = 6


    @staticmethod
    def auto_adjust_CVputTextY(text_pos_y : int, txt_h : int, baseline : int, ) -> int:
        """
        text_posy :  the one supplied to cv2puttext org
        txt_h, baseline from self.get_txt_WHBaseline
        """
    
        if text_pos_y < txt_h:
            # this ensures text too high are visible
            text_pos_y = txt_h

        else:
            # this allows us to supply a desired end pos y for the txt including the baseline; 
            # it return an updated pos that accounts for the baseline
            text_pos_y = text_pos_y - baseline
        
        return text_pos_y

    def get_txt_WHBaseline(self, text: str, fontface = PO_CONST.FONTFACE, 
                           fontscale = PO_CONST.FONTSCALE, fonthick = PO_CONST.FONTTHICK) -> Tuple[int, int, int]:
        
        (txt_w, txt_h), baseline = cv2.getTextSize(text, fontface, fontscale, fonthick) 

        return txt_w, txt_h, baseline

    def rectangle(self, frame, p1, p2, color=None, bg_color = None):
        if not color:
            color = self.color
        if not bg_color:
            bg_color = self.bg_color

        cv2.rectangle(frame, p1, p2, bg_color, 3)
        cv2.rectangle(frame, p1, p2, color, 1)

    def put_depth(self, frame, point, depth):
        # depth expected in mm
        cv2.circle(frame, point, 10, (0,0,255))
        cv2.putText(frame, "{}cm".format(depth/10), (point[0],point[1]), self.text_type, 1 ,  (0,0,255) , 2 )

    def put_ROIinfo(self, frame : npt.NDArray[np.uint8], roi_data : Dict): 
        
        x = frame.copy()

        for roi in roi_data.values():
  
            spatials = roi["spatials"]

            self.rectangle(x, (roi["xmin"], roi["ymin"]), (roi["xmax"], roi["ymax"])) 
            self.rectangle(x, (roi["xmin3"], roi["ymin3"]), (roi["xmax3"], roi["ymax3"])) 

            self.putText(x, "X: " + ("{:.1f}cm".format(spatials['x']/10) if not math.isnan(spatials['x']) else "--"), (roi["cx"] + 10, roi["cy"] + 20))
            self.putText(x, "Y: " + ("{:.1f}cm".format(spatials['y']/10) if not math.isnan(spatials['y']) else "--"), (roi["cx"] + 10, roi["cy"] + 35))
            self.putText(x, "Z: " + ("{:.1f}cm".format(spatials['z']/10) if not math.isnan(spatials['z']) else "--"), (roi["cx"] + 10, roi["cy"] + 50))

        return x
        
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)

    def putText_mid(self, frame, text, FONTFACE = None, FONTSCALE = None, FONTTHICK = None, font_color = None):
        if FONTFACE is None:
            FONTFACE = self.text_type
        if FONTSCALE is None:
            FONTSCALE = self.FONTSCALE
        if FONTTHICK is None:
            FONTTHICK = self.FONTTHICK
        if font_color is None:
            font_color = self.color

        label_width, label_height, baseline = self.get_txt_WHBaseline(text, FONTFACE, FONTSCALE, FONTTHICK)
        cv2.putText(frame, text, (int((frame.shape[1]-label_width)/2),int((frame.shape[0]-label_height)/2 + baseline)),
                     FONTFACE, FONTSCALE, font_color, FONTTHICK)
        
    def putText_top(self, frame, text, FONTFACE, FONTSCALE, FONTTHICK, font_color):
        label_width, label_height, baseline = self.get_txt_WHBaseline(text, FONTFACE, FONTSCALE, FONTTHICK)
        cv2.putText(frame, text, (int((frame.shape[1]-label_width)/2),int(label_height+ baseline)),
                     FONTFACE, FONTSCALE, font_color, FONTTHICK)

@define(slots=True, frozen=True, kw_only=True)
class Frame:
    bgrframe : np.ndarray = field(factory=lambda: np.ndarray(shape=(0,0,3), dtype=int))
    resized_bgr : np.ndarray = field(factory=lambda: np.ndarray(shape=(0,0,3), dtype=int))
    static_depthmsg : np.ndarray = field(factory=lambda: np.ndarray(shape=(0,0,3), dtype=int))
    depthFrame : np.ndarray = field(factory=lambda: np.ndarray(shape=(0,0,3), dtype=int))

@define(slots=True, frozen=True)
class bbox_XY():
    #! do not change the order of attributes; may hv codes that unpack a list inside here without using kwargs
    xmin = field(factory=int)
    ymin = field(factory=int)
    xmax = field(factory=int)
    ymax = field(factory=int)

def calc_fps(cam_start_time : float) -> float: # @ last_update 04 jun 2023

    display_end_time = time.perf_counter() 
    display_fps = str(round(1 / (display_end_time - cam_start_time )))
    
    return display_fps

def calc_timeElapsed(t_start : float) -> float: # @ last_update 04 jun 2023
    elapsed_time = time.perf_counter() - t_start
    
    return elapsed_time

def clamp(num, v0, v1): # @ last_update 04 jun 2023
    
    return max(v0, min(num, v1))

def containsLetterAndNumber(x : str) -> bool:
    """
    checks whether a string is strictly alphanumeric

    returns
        False if does not contain or not just alphanumeric
    """
    # from https://stackoverflow.com/questions/64862663/how-to-check-if-a-string-is-strictly-contains-both-letters-and-numbers
    return x.isalnum() and not x.isalpha() and not x.isdigit()

def count_occurence(input_list : List, text : str) -> int: # @ last_update 04 jun 2023
    """
    returns number of times `text` occured in list
    """

    counts = input_list.count(text)

    return counts

def count_occurencev2(search_for : List[str], search_from: List[str]) -> Dict[str, int]:
    counter = Counter(search_from)
    return {string: counter[string] for string in search_for}


def cv2_custom_quit(cv2_windowname : str):
    break_loop = False

    key = cv2.waitKey(10)        
    if cv2.getWindowProperty(cv2_windowname, cv2.WND_PROP_VISIBLE) < 1:
        break_loop = True
    elif key in [ord('q'),27] :
        cv2.destroyWindow(cv2_windowname)
        break_loop = True
    
    return break_loop

def cv2show(cv2_win_name : str = PO_CONST.CV_WIN_NAME.DEBUG,
            img : Optional[npt.NDArray[np.int_]] = None):

    cv2.namedWindow(cv2_win_name) 
    cv2.imshow(cv2_win_name, img)  
    cv2.waitKey(1) 

def devordebug(SES_SETTING):
    
    if SES_SETTING["session"]["DEV_MODE"] or SES_SETTING["session"]["debug"]:
        return True
    else:
        return False

def df_to_clipboard(df : pd.DataFrame , n : int = 0) -> None:
    """
    copies df to clipboard with correct indentations
    
    n : number of indentations
    """
    indent = " " * n
    df_str = indent + df.to_string().replace("\n", "\n" + indent)
    
    pd.DataFrame([df_str]).to_clipboard(index=False, header=False)

def flatten_dict(nested_dict) -> Dict: # @ last_update 04 jun 2023
    #https://stackoverflow.com/questions/12118695/efficient-way-to-remove-keys-with-empty-strings-from-a-dict?answertab=modifieddesc#tab-top
    
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res

def get_system_time(merge : bool = False) -> Union[Tuple[str,str], str]: # @ last_update 04 JUN 2023
    """"
    returns system datetime
    """
    now = datetime.datetime.now() 
    now_date = datetime.datetime.date(now)
    time_now = now.strftime("%H%M%S")
    date_now = now_date.strftime("%Y%m%d")

    if not merge:
        return date_now, time_now
    else:
        return str(date_now + "_" + time_now)
    
def get_exception_info() :
    exception_info_dict = {}
    info_str = ""

    try:
        exception_type, exception_object, exception_traceback = sys.exc_info()

        exception_info_dict["exception_type"] = exception_type
        exception_info_dict["filename"] = exception_traceback.tb_frame.f_code.co_filename
        exception_info_dict["line_number"] = exception_traceback.tb_lineno
         
        for i, (key, value) in enumerate(exception_info_dict.items()):
            info_str += "\n"

            if i > 0:
                info_str += "\n" 

            info_str += f"{key}: {value}"

        try:
            info_str += f"\n{traceback.format_exc()}"
        except:
            pass

    except:
        pass

    return info_str

def indices(input_list : List , target : Any) -> List[int]:
    """
    eg indices(["apple","apple","salt","grape"],"apple")
        gives [0, 1]
        indices(["apple","apple","salt","grape"],"sa")
        gives []
    """

    indices = [ind for ind, ele in enumerate(input_list) if ele == target]

    return indices

def isSerialNo(x: str) -> bool: # @ update 13 jun 2023
    return x[0:2] in PO_CONST.COUNTRY_ALPHA2_CODE and len(x)==10 and any(char.isdigit() for char in x)

def isModelNumber(x: str) -> bool: # @ update 13 jun 2023
    return 5<=len(x)<8 and containsLetterAndNumber(x) and len(''.join(filter(str.isdigit, x)))>=4

def isOptionLabel(x: str) -> bool:
    return len(x)==6 and x[0:3]=="Opt"

def isOptionNumber(x: str) -> bool:
    return len(x)>= 3 and len(x) % 3 == 0

def isCountry(x : str) -> bool:
    return x.capitalize() in PO_CONST.COUNTRIES

def isUnique(s : pd.Series) -> bool:
    """
    Converts a pandas series into numpy and check that all elements == first element

    """
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def nested_dict_to_df(values_dict) -> pd.DataFrame:
    """
    https://stackoverflow.com/questions/12118695/efficient-way-to-remove-keys-with-empty-strings-from-a-dict?answertab=modifieddesc#tab-top

        
    use to convert label_with_missing_or_incorrect_data into a more readable pandas
    capable of handling irregular nested dict
    example usage:
    so = {'Lspo': {12: {'no_of_missing_data': 1}},
     'Lptd': {11: {'no_of_missing_data': 3}},
     'Lst': {6: {'missing_data': [0]},
      7: {'missing_data': [1]},
      8: {'missing_data': [0, 1]},
      9: {'missing_data': [0, 1]},
      10: {'missing_data': [0, 1], "incorrect_num": 42}},
     'Lkcc': {5: {'label_vs_config': False}}}

    xxx = nested_dict_to_df(so)
    xxx.fillna('', inplace=True)
    """
    
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    
    return df

def pyplt_img(img, figsz = (10,10), gray=False, title=""):
    try:  
        plt.figure(figsize=figsz)
        
        if img.ndim == 3:
            plt.imshow(img)
        elif img.ndim == 2:
            plt.imshow(img, cmap="gray", vmin=0, vmax=255) 
            
        plt.title(title)
        plt.show() 

    except Exception as e:
        gui.popup(f"Error when plotting image. Error:\n{e}")

def is_notebook() -> bool:
    """
    https://stackoverflow.com/a/39662359
    """
    try:
        shell = get_ipython().__class__.__name__ 
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def disp_or_print_df(df : Union[pd.DataFrame, Dict[str, pd.DataFrame]] , title : str = "", 
                     side_by_side : bool = False) -> None:

    if is_notebook():
        if isinstance(df, pd.DataFrame):
            df_styler = df.style.set_table_attributes("style='display:inline'").set_caption(title)
            display_html(df_styler._repr_html_(), raw = True)

        if isinstance(df, dict):

            if side_by_side:
                df_styler = []

                for df_caption, dfs in df.items():
                    df_styler.append(dfs.style.set_table_attributes("style='display:inline; margin-right:50px;'").set_caption(df_caption))
                
                display_html("".join([x._repr_html_() for x in df_styler]), raw=True)

            else:
                for caption , dfs in df.items():
                    df_styler = dfs.style.set_table_attributes("style='display:inline'").set_caption(caption)
                    display_html(df_styler._repr_html_(), raw = True)
                    
    else:
        if isinstance(df, pd.DataFrame):
            print(f"\n{title}")
            pprint(df)
        
        if isinstance(df, dict):
            for caption , dfs in df.items():
                print(f"\n{caption}")
                pprint(dfs)

def pyplot_subplots(imgs : List[npt.NDArray], titles : List[str],  
                    figsize : Tuple[int, int] = (10,10), col_row : Tuple[int, int] = (2,1)) -> None:
    
    if is_notebook():
        plt.figure(figsize=figsize)
        for number in range(len(imgs)):
            ax = plt.subplot(col_row[1], col_row[0], number+1)
            ax.set_title(titles[number])
            plt.imshow(imgs[number])
        plt.tight_layout
        plt.show()

def return_as_list(x : Union[Any, List[Any]]) -> List[Any]: # @ last_update 04 jun 2023
    if not isinstance(x,list):
        x = [x]

    return x

def resize_img(image : npt.NDArray[np.int_], 
                width : Optional[int] = None, 
                height : Optional[int] = None, inter = cv2.INTER_AREA) -> npt.NDArray[np.int_]:
    """ 
    recursive function that resize image with open cv without distortion
    # https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    
    variables:
    image: image

    width: desired outcome width

    height: desired outcome height

    inter: open cv interpolation to resize img. default: INTER_AREA

    returns: 
    resized: 
        resized image; 
        original image (if width and height was not specified)
    """
    
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, return original image 
    if width is None and height is None:
        return image

    if width != w or height != h:
        if width is None:
            # calculate the ratio of the height and construct the dimensions 
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation = inter)

        return resized
    else:
        return image

def scrub_dict(d : Dict[Any, Any]) -> Dict[Any, Any]:
    """
    # https://stackoverflow.com/questions/12118695/efficient-way-to-remove-keys-with-empty-strings-from-a-dict?answertab=modifieddesc#tab-top
    
    function:
        not removing None, {}, [] from nested dictionary 
        for use after labels checking
    
    input: nested dict

    returns: scrubbed dict
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = scrub_dict(v) 
        if v not in (None, {}, []):
            new_dict[k] = v
            
    return new_dict

def string_lower_1stchar(x : str) -> str:
    
    if len(x) > 0:
        return x[0].lower() + x[1:]
    else:
        return x
    
def string_replace_endperiod2comma(x : str) -> str:
    if len(x)>0 and x.endswith("."):
        return x[:-1] + ","
    else:
        return x

def get_dist_cam2subject(subject_name: str, preset = PO_CONST.COBOT_CAM_2SUBJECT) -> float:
    """
    get the distance between color cam and subject
    
    """
    try:
        x = preset[subject_name][PO_CONST.DIST_CAM2SUB] 
    
    except KeyError:
        while True:
            x : str = gui.popup_get_text(f'{subject_name}: dist_cam2sub not set.\
                                                \nEnter dist_cam2sub [mm]:')
            if x:
                x : Optional[re.Match] = re.search(r"-?(?:\d+\.?\d*|\.\d+)", x) # returns 

                try:
                    x = float(x[0])
                    break
                except TypeError:
                    # catches None
                    pass
    return x

def mask_region_oustideROI(img : npt.NDArray[np.uint8], roi_xy : Union[List[bbox_XY], bbox_XY], 
                           tolx : int = 10, toly : int = 10) -> npt.NDArray[np.uint8]:

    h,w = img.shape[:2]
    
    mask = np.zeros((h,w), dtype=np.uint8)

    roi_xy = return_as_list(roi_xy)
    
    for roi in roi_xy:
        ymin = clamp(roi.ymin-toly,0,roi.ymin)
        ymax = clamp(roi.ymax+toly,roi.ymax,h)
        xmin = clamp(roi.xmin-tolx,0,roi.xmin)
        xmax = clamp(roi.xmax+tolx,roi.xmax,w)
  
        mask[ymin:ymax,xmin:xmax]=1
    
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img[np.where(mask == 0)] = [0, 0, 0] # required for multichannel img accroding to GPT4

    return masked_img

def calc_bbox_L2norm_wrt_img_center(df : pd.DataFrame, img : npt.NDArray[np.uint8]) -> None:
    """
    L2 norm aka euclidean distance
    """
    
    h, w = img.shape[:2]
    
    # find centroid of bbox
    df["cx"] = ((df["xmin"] + df["xmax"]) / 2).astype(int) 
    df["cy"] = ((df["ymin"] + df["ymax"]) / 2).astype(int)

    # calc euclidean dist from bbox centroid to img centroid
    df["eucli_d"] = np.sqrt((h/2 - df["cy"])**2 + (w/2 - df["cx"])**2)


def filter_objs_without_intersection(df : pd.DataFrame, parent_objs : Union[str, List[str]],
                                                   obj_to_not_exclude : List[str] = ["Lptd", "cd", "cb"]):
    """
    #! Warning, don't think we should exclude "cd", "cb"

    - filter obj detected that doesn't intersect with parent_objs
    - objects in `obj_to_not_exclude` are not removed if they don't intersect with parent_objs

    Args
        parent_objs: typically "name" of products in obj detection which labels/components are installed on
    
    Example ussage
    data = {'name': ['t0', 'Lptd', 't2', 't3', 't0'], 
        'xmin': [10, 20, 15, 55, 50], 'ymin': [10, 20, 15, 55, 50], 
        'xmax': [20, 30, 25, 65, 60], 'ymax': [20, 30, 25, 65, 60]}
    df = pd.DataFrame(data, index = [0,5,6,9,8]) # simulates non-ordered index
    filter_objs_without_intersection(df, ["t0"], ["Lptd"])
    """
    BBOX_TEMPLATE = asdict(bbox_XY()).keys()

    # index, bbox of all objs in tensor format
    index = torch.tensor(df.index.values)
    bboxs = torch.tensor(df[BBOX_TEMPLATE].values)

    parent_objs = return_as_list(parent_objs)

    isParentObj : pd.Series = df['name'].isin(parent_objs)
    parent_objs_idx : List[int] = df.loc[isParentObj].index.tolist()
    parent_objs_bbox = torch.tensor(df.loc[isParentObj, BBOX_TEMPLATE].values)   

    mask_of_labels_forcefully_included = torch.tensor(df['name'].isin(obj_to_not_exclude).values)
    
    """
    sample `iou` with 2 targets (ie 2 rows): the cols contains targets and other objs

    tensor(
        [[1.00000, 0.00000, 0.14286, 0.00000, 0.00000],
        [0.00000, 0.00000, 0.00000, 0.14286, 1.00000]])

    intersection_mask -> True if subject intersects with >=1 parent_obj(s)
        tensor([ True, False,  True,  True,  True])

    intersection_index = index from df which intersection mask is True
        [0, 2, 3, 4]

    remove df_idx of targets from  intersection_index
        [2, 3]

    """

    iou = box_iou(parent_objs_bbox, bboxs)

    if iou.numel(): # this means it's not empty
        # mask out objs that intersects with parent, 
        # included those that are focefully included even if they dont intersect with any parent
        intersection_mask : torch.Tensor = (iou.max(dim=0).values > 0) | mask_of_labels_forcefully_included

    else:
        intersection_mask : torch.Tensor = mask_of_labels_forcefully_included

    intersection_index  : list[int] = index[intersection_mask].tolist()
    intersection_index = [i for i in intersection_index if i not in parent_objs_idx]

    if len(intersection_index) == 0:
        logger.info(f"No obj intersects wit parent objs : {parent_objs}")
    else:
        logger.info(f"Index of rows with intersections: {intersection_index}")
        
        for i in intersection_index:
            target_iou = iou[:, index.eq(i)].squeeze()
            
            max_iou = target_iou.max(dim=0).values.item()
            max_index = target_iou.argmax(dim=0).item()

            parent_obj_with_max_iou = parent_objs_idx[max_index]

            _msg = f"""{df.at[i, 'name']} at idx {i} intersect 
                        {df.at[parent_obj_with_max_iou, 'name']} 
                        at idx {parent_obj_with_max_iou}, max iou : {max_iou:.3f}"""

            logger.info(' '.join(_msg.split()))

    return df.drop(index[~intersection_mask])