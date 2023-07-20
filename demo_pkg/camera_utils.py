
import PySimpleGUI as gui
from typing import Optional
import cv2
import numpy as np
import numpy.typing as npt
from attrs import define, field

from .polaris_logger import logger
from . import POLARIS_CONSTANTS as PO_CONST
from . import polaris_utils as po_utils
from .guimaker import GUIMaker


@define(slots=True)
class PredefinedAFRegion():
    
    """
    h, w refers to 12MP reso of Oak-D pro
    """
    h : int 
    w : int 
    
    default : po_utils.bbox_XY = field(init=False)
    chassis_frnt_Lst : po_utils.bbox_XY = field(init=False)
    
    def __attrs_post_init__(self):
        # focuses on a region close to centre of img
        self.default = po_utils.bbox_XY(int(self.w/4) , int(self.h/4),
                                        int(self.w/4*3), int(self.h/4*3))
        
        # for use in closeup for chassis with LST on right side of chassis , when using this the closeup lst is on the left of img
        self.chassis_frnt_Lst = po_utils.bbox_XY(int(self.w/4), int(self.h/4), 
                                                 int(self.w/4+200),int(self.h/4*3))

class AFRegionUtil:

    def __init__(self, AFROI : Optional[po_utils.bbox_XY] = None , request_roi : bool = False):  
        """
        request_roi : triggers a gui to ask for afroi region
        """     
        if request_roi:
            self.AFROI = GUIMaker().ask_xy_for_AFonROI() 
        else: 
            self.AFROI = AFROI
            if not self.AFROI:
                message = "AFROI is None, this will lead to errors."
                logger.critical(message)
                gui.popup(message)

        self.step = 1000
        self.position = (self.AFROI.xmin, self.AFROI.ymin)
        self.size = (self.AFROI.xmax - self.AFROI.xmin, self.AFROI.ymax - self.AFROI.ymin)
        
        # this was done in according to OAKD tutorial. they used camrgb.getResolutionSize which follows the (w, h;
        # here we reverse the constant defined in (h,w) to (w,h)
        self.resolution = PO_CONST.CLR_CAM_RESO[::-1] 
        self.maxDims = self.resolution[0], self.resolution[1]

    def toRoi(self):
        roi = np.array([*self.position, *self.size])  
        return roi 
    
    def display_region(self, name : str, frame : npt.NDArray):
        """
        display afroi region
        """
        _frame = frame.copy()
        
        cv2.rectangle(_frame, self.position, self.endPosition(), (0, 255, 0), 1)
        cv2.rectangle(_frame, self.position, self.endPosition(), (0, 255, 0), 3)
        po_utils.cv2show(name, cv2.resize(_frame, PO_CONST.FRAME_DISP_SZ))
        
    def move(self, x=0, y=0):
        self.position = (
            po_utils.clamp(x + self.position[0], 0, self.maxDims[0]),
            po_utils.clamp(y + self.position[1], 0, self.maxDims[1])
        )

    def endPosition(self):
        return (
            po_utils.clamp(self.position[0] + self.size[0], 0, self.maxDims[0]),
            po_utils.clamp(self.position[1] + self.size[1], 0, self.maxDims[1]),
        )
