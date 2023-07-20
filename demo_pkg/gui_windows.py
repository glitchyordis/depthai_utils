import PySimpleGUI as gui
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List, Optional, TYPE_CHECKING, Any

from . import POLARIS_CONSTANTS as PO_CONST
from . import polaris_logger
from . import polaris_utils as po_utils

import pyautogui
import multiprocessing as mp
import yaml

logger = polaris_logger.logger

with open(PO_CONST.SES_SETTING_YAML, "r") as stream:
    SES_SETTING = list(yaml.safe_load_all(stream))[0]

devordebug = po_utils.devordebug(SES_SETTING)

class GUIUtils():
    def __init__(self) -> None:
        pass

    def out_msg2term(self, message=None, t_color=None, bg_color=None, lvl = None, 
                                exp_msg = None, auto_newline = True, 
                                autolog : Union[bool, Tuple[bool, str]] = (True, "info"),
                                popup: Union[bool, Tuple[bool, str]] = (False, "Warning")): # @ last_update 04 jun 2023 
        
        """
        autolog: #! if want to use autolog, pass tuple as input or dont pass in keyword arg to use default setting
                #! if dont want to log pass in anything tat's not a tuple or tuple("false", <>) . we recommend just pass in False
        popup: #! pass tuple with True to enable popup or pass in bool True to use default popup title
        
        """
        
        try:
            if auto_newline: print("\n", end="") 

            if isinstance(self.main_win, gui.Window): 
                
                def gui_mprint(*args, **kwargs):
                    self.main_win['-OUTPUT-' + gui.WRITE_ONLY_KEY].print(*args, **kwargs)
                    self.main_win['-OUTPUT2-' + gui.WRITE_ONLY_KEY].print(*args, **kwargs)

                msg_lvl_color = {"red":{"t_color":"white", "bg_color":"red"},
                                "g":{"t_color":"white", "bg_color":"green"},
                                "o":{"t_color":"white", "bg_color":"#ff9500"},
                                "purple":{"t_color":"white", "bg_color":"purple"}}

                if lvl:
                    gui_mprint(message, t=msg_lvl_color[lvl]["t_color"], b=msg_lvl_color[lvl]["bg_color"])
                else:
                    gui_mprint(message, t=t_color, b=bg_color)

            else:
                # if autolog is not a tuple and tuple[0] is not True
                if not (isinstance(autolog, tuple) and autolog[0]): 
                    # enables printing the similar msg in jupyter if gui not initialized
                    if not exp_msg:
                        print(message)
                    elif exp_msg == "RPT":
                        print(f"{'#' * 11} {message} {'#' * 11}")
                    else:
                        print(exp_msg)
                        
            if isinstance(autolog, tuple): 
                if autolog[0]:
                    log_method = polaris_logger.LOG_LEVEL_MAPPING.get(autolog[1], None)

                    if log_method:
                        log_method(message) 
                
            if isinstance(popup, tuple):
                if popup[0]:
                    gui.popup(message, title = popup[1])
            elif popup:
                gui.popup(message, title = "Warning")

        except Exception as e:
            logger.exception(f" Error at gui_output_msg2terminal.")
            gui.popup(f"gui_output_msg2terminal. Error:\n{e}\nMessage to display:\n{message}")

    def out_presetmsg(self, preset_name : str): # @ last_update 04 jun 2023

        preset = PO_CONST.PRESET_MSG.get(preset_name, 
                                         PO_CONST.PRESET_MSG["preset_msg_error"])

        self.out_msg2term(message=preset.get("message", None),
                                t_color=preset.get("t_color", None), 
                                bg_color=preset.get("bg_color",None), 
                                lvl = preset.get("lvl", None), 
                                exp_msg = preset.get("exp_msg",None),
                                auto_newline = preset.get("auto_newline", True),
                                autolog = preset.get("autolog", (True, "info")),
                                popup = preset.get("popup", (False, "Warning")))

    def f_btn_pressed(self, preset_name : str, windows_name : Union[str, List[str]] = "main"):
        
        self.update_valsNcolor(windows_name=windows_name, 
                               element_key='-OUTPUT2-'+gui.WRITE_ONLY_KEY, value="", 
                               force=("value"))
        
        self.out_presetmsg(preset_name)

    def f_AF(self, focusMode : str): # @ last_update 06 jun 2023
        key = "-lens_position-"
        self.update_valsNcolor(element_key=key, value= "AF mode")

        key = '-autofocus_combo-'
        self.update_valsNcolor(element_key=key, value = focusMode)
        
    def f_AFv2(self, focusMode: str, k1 : Tuple[bool, str] = (True, "AF mode")):
        
        if k1[0]:
            key = "-lens_position-"
            self.update_valsNcolor(element_key=key, value = "AF mode")

        key = '-autofocus_combo-'
        self.update_valsNcolor(element_key=key, value = focusMode)
        
         

    def get_valid_active_win(self, windows_name : Union[str, List[str]] = "main") -> Dict[str, gui.Window]: # @ last_update 04 jun 2023
        
        available_wins = self.get_all_win()

        if windows_name == "all":

            active_wins = {win_key: window for win_key, window in available_wins.items() if isinstance(window, gui.Window)}
        else:
            windows_name = po_utils.return_as_list(windows_name)

            windows_name = [windows for windows in windows_name if windows in available_wins.keys()]

            active_wins = {windows: available_wins[windows] for windows in windows_name if isinstance(available_wins[windows], gui.Window)}

        return active_wins

    def enable_disable_btn(self, target_btns : dict[str,List[str]], windows_name : Union[str, List[str]] = "main") -> None: # @ last_update 04 jun 2023

        active_windows = self.get_valid_active_win(windows_name)

        for window in active_windows.values():
            for actions, buttons in target_btns.items():
                if actions == "disable":
                    for button in buttons:
                        if button in window.key_dict:
                            window[button].update(disabled=True)
                elif actions == "enable":
                    for button in buttons:
                        if button in window.key_dict:
                            window[button].update(disabled=False) 

    def enable_disable_allbtns(self, windows_name : Union[str, List[str]] = "all", mode = "enable") -> None: # @ last_update 04 jun 2023
                                      
        active_wins = self.get_valid_active_win(windows_name)

        for win_key, win in active_wins.items():
            btns = [k for k, obj in win.key_dict.items() if isinstance(obj,gui.B)]
            self.enable_disable_btn({mode : btns}, [win_key])

    # @ last_update 04 jun 2023
    def update_valsNcolor(self, element_key : str, 
                                data : Any = None,
                                value : Optional[str] = None, 
                                vals : Optional[str] = None, 
                                text : Optional[str] = None, 
                                change_background : Tuple[bool, Optional[str]] = (False,None),
                                button_color: Tuple[bool, Optional[str]] = (False,None),
                                windows_name : Union[str, List[str]] = "main",
                                force : Tuple[str] = ()): 

        """
        updates gui element's value/values &/or background_color

        Args
            value vs vals: 
                some element, when performing update uses update(value = ?) while others use update(values = ?) 
                hvent seen any that uses both

            # ! force
            with the current implementation vals, value, text, if passed in as None or "" only updates if their headings is in force
        """
        active_windows = self.get_valid_active_win(windows_name)
        X = [0, False]
        
        for win in active_windows.values():
            if element_key in win.key_dict:
                if vals or vals in X:
                    win[element_key].update(values=vals)
                else:
                    if "vals" in force:
                        win[element_key].update(values=vals)
                
                if value or value in X:
                    win[element_key].update(value=value)
                else:
                    if "value" in force:
                        win[element_key].update(value=value)

                if text or text in X:
                    win[element_key].update(text=text)
                else:
                    if "text" in force:
                        win[element_key].update(text=text)

                if data:
                    win[element_key].update(data=data)
                else:
                    if "data" in force:
                        win[element_key].update(data=data)

                if change_background[0]:
                    win[element_key].update(background_color = change_background[1])

                if button_color[0]:
                    win[element_key].update(button_color = button_color[1])


@dataclass
class GUIWindows(GUIUtils): # @ last_update 04 jun 2023
    values = None
    event = None
    window = None
    
    main_win : Optional[gui.Window] = None # main window
    side_win : Optional[gui.Window] = None
    issue_log_win : Optional[gui.Window] = None
    label_issue_fix_win : Optional[gui.Window] = None

    cobotqns : Optional[object] = None # cannot import object here since circular import causes vscode to unable to detect unit tests

    if not devordebug:
        def read_all(self):
            self.window, self.event, self.values = gui.read_all_windows(timeout=5)
    else:
        def read_all(self):
            self.window, self.event, self.values = gui.read_all_windows(timeout=5)

            if self.event and self.event!= "__TIMEOUT__":
                self.enable_disable_allbtns(mode = "enable")

    def check_exit(self):
        break_loop = False

        if self.event == 'Cancel' or self.event == gui.WIN_CLOSED:  
            # self.window.close()
            if self.window == self.issue_log_win: 
                self.window.close()
                self.issue_log_win = None
                
            elif self.window == self.label_issue_fix_win:
                self.window.close()
                self.label_issue_fix_win = None

            elif self.window == self.main_win:
                if self.cobotqns is not None and \
                    isinstance(self.cobotqns.p_cobot, mp.Process) and self.cobotqns.p_cobot.is_alive():
                    self.cobotqns.q_cobot_job.put(None)

                break_loop = True

        return break_loop


    def get_all_win(self):
        return {"main" : self.main_win, 
                "side" : self.side_win}
    
    def set_main_gui_pos(self):
        screen_w, screen_h = pyautogui.size()

        if isinstance(self.main_win, gui.Window):
            win_w, win_h = self.main_win.size

            x_pos, y_pos = screen_w - win_w, 0

            self.main_win.move(x_pos, y_pos)