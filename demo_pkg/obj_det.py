import PySimpleGUI as gui
from typing import Dict, Tuple, List, Optional
from . import POLARIS_CONSTANTS as PO_CONST
from . import polaris_utils as po_utils
import yaml
import pandas as pd
from .polaris_logger import logger
import numpy as np
from numpy import typing as npt
import cv2
import torch

# get session setting
with open(PO_CONST.SES_SETTING_YAML, "r") as stream:
    SES_SETTING : Dict = list(yaml.safe_load_all(stream))[0]

gui_mode = SES_SETTING["session"]["gui_mode"]

class YOLOObjDetUtils():
    def __init__(self):
        pass
    
    # @ update 08 jun 2023
    def obj_det_getResult(self, img : npt.NDArray[np.uint8] , size : int = 832) -> Tuple[object, pd.DataFrame]:
                          
        """
        perform object detection on img supplied.
                
        var:
            img:        image
            size:       image size, pytorch hub var; imgsz to model will use to detect objs

        default outputs and datatypes from yolo
            xmin          float64
            ymin          float64
            xmax          float64
            ymax          float64
            confidence    float64
            class           int64
            name           object

        after conversion
            xmin            int32
            ymin            int32
            xmax            int32
            ymax            int32
            confidence    float64
            class           int64
            name           object
            dtype: object
        """ 

        result = self.model(img, size = size)

        # det result
        data : pd.DataFrame = result.pandas().xyxy[0]
        data = data.astype({"xmin": "int", "ymin": "int", "xmax":"int", "ymax":"int"})
                            
        logger.info("\nLabelObjDet result:" + \
                    '\n\t'+ data.to_string().replace('\n', '\n\t'))
        
        if not gui_mode: 
            print(f"data:\n{data}")

        return result, data
    
    def plot_subjects(self, ori_img : npt.NDArray[np.uint8], df : pd.DataFrame):
        FS =  PO_CONST.FONTSCALE
        FT =  PO_CONST.FONTTHICK
        FF = PO_CONST.FONTFACE

        thelper = po_utils.TextHelper()

        img = ori_img.copy()

        for _, row in df.iterrows():
            clr = PO_CONST.HEX2RGB[row["class"]]

            xmin = row["xmin"]
            ymin = row["ymin"]
            xmax = row["xmax"]
            ymax = row["ymax"]

            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1 = pt1, pt2 = pt2 , color = clr, thickness = 10)

            obj_info = str(row["name"]) + " " + str(round(row["confidence"],2))
            txt_w, txt_h, baseline = thelper.get_txt_WHBaseline(obj_info)
            text_pos_y = thelper.auto_adjust_CVputTextY(ymin, txt_h, baseline)
            cv2.rectangle(img, pt1 = (xmin , text_pos_y - txt_h - FT) , 
                      pt2 = (xmin + txt_w + FT, ymin),
                      color = clr, thickness = -1)
            cv2.putText(img, obj_info, (xmin + FT, text_pos_y) , 
                        FF, FS, (255,255,255), FT)

        return po_utils.resize_img(img, width=800)

    
        # if not po_utils.is_notebook():
        #     title = PO_CONST.CV_WIN_NAME.MAIN
        #     po_utils.cv2show(cv2_win_name = PO_CONST.CV_WIN_NAME.MAIN,
        #                      img = resized_img)
        #     while True:
        #         if po_utils.cv2_custom_quit(title):
        #             break
        # else:
        #     title = PO_CONST.CV_WIN_NAME.OBJDET
        #     po_utils.pyplt_img(resized_img, title = title)

class LabelObjDet(YOLOObjDetUtils):

    def __init__(self):
        setting = SES_SETTING["label_obj_det"]

        self.model : Optional[torch.nn.Module] = None
        self.DETECTOR_TARGETS : List[str] = []
        self.output_mapper : Dict[str, int] = {}
        
        # get a list of objects the algorithm was trained to detect without loading model
        with open(setting["yaml_file"], "r") as stream:
            try:
                self.DETECTOR_TARGETS = list(yaml.safe_load_all(stream))[0]["names"]
                
                # maps each DECTOR_TARGETS to a number eg output_mapper = {'Lccr': 0, 'Lce': 1 .....}
                self.output_mapper =  dict(zip(self.DETECTOR_TARGETS,[i for i in range(len(self.DETECTOR_TARGETS))]))

            except Exception:
                message = f"Failed to load label obj det YAML file. {PO_CONST.CBPPME}"
                gui.popup(message, title = "Fatal")
                logger.critical(message)
                logger.exception()
                
        # load model
        if SES_SETTING["session"]["obj_det_enabled"]:
            
            if torch.cuda.is_available():
                self.model = torch.hub.load('ultralytics/yolov5', 
                                        'custom', 
                                        path=setting["pt_file"])
                
                # class names to ignore
                if "ignore" in setting:
                    self.model.classes = [cls for cls, name in self.model.names.items() if \
                                        name not in setting["ignore"]]

                if "conf" in setting:
                    self.model.conf = setting["conf"]
            else:
                message = f"Torch cuda is not available. {PO_CONST.CBPPME}"
                gui.popup(message, title = "Fatal")
                logger.critical(message)