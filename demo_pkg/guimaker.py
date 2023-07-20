import PySimpleGUI as gui
from typing import Tuple, List, Optional
import yaml
from distutils.util import strtobool
import pathlib

from .polaris_logger import logger
from . import POLARIS_CONSTANTS as PO_CONST
from . import polaris_utils as po_utils

POLRS = "Polaris"

with open(PO_CONST.SES_SETTING_YAML, "r") as stream:
    SES_SETTING = list(yaml.safe_load_all(stream))[0]

class GUIMaker:

    def __init__(self) -> None:

        self.GENERAL_FONT = "SegoeUI "
        self.TERMINAL_FONT = "CascadiaCode"
        self.HEADER_FONT_SZ = "18"
        self.TITLE_FONT_SZ = "16"
        self.BODY_FONT_SZ = "12"
        self.CAPTION_FONT_SZ = "9"
        self.BOLD = " bold" 
        self.GF_CFSZ = self.GENERAL_FONT + self.CAPTION_FONT_SZ

        gui.set_options(font=self.GENERAL_FONT +  self.BODY_FONT_SZ)

        # gui.theme('Topanga')      # Add some color to the window
        # gui.theme_bg_color("#F2F2F7")
        # gui.theme_button_color(("#000000","#FFFFFF"))
        # gui.theme_text_element_bg_color("#E5E5EA") 
        # gui.theme_input_bg_color("#FFFFFF")
        # gui.theme_input_text_color("#000000")
        # gui.theme_text_color("#000000")

        ## crosschecking between CoC vs PTD vs Serial tag will be displayed in a table in GUI
        ## this define the table properties
        # checker_result_heading = ['index', 'ModelNo', 'SerialNo', 'Options', 'TempOptions']

        self.checker_result_heading = ['index',"name", 'ModelNo', 'SerialNo', 'Options', 'TempOptions']
        self.checker_result_col_widths = list(map(lambda x:len(x)+2, self.checker_result_heading)) # find the widths of columns in character.
        self.checker_result_font = ('Courier New', 11)
        self.checker_result_max_col_width = len('ParameterNameToLongToFitIntoAColumn')+2  # Set max midth of all columns of data to show
        self.char_width = gui.Text.char_width_in_pixels(self.checker_result_font)     # Get character width in pixel

        # self.checker_result_layout = [
        #     [gui.Table(values = [],
        #     headings = self.checker_result_heading,
        #     max_col_width=66,
        #     auto_size_columns = False,
        #     display_row_numbers=False,
        #     def_col_width=16,
        #     justification = "right",
        #     num_rows=10,
        #     key = "-Table-",
        #     row_height = 35,
        #     col_widths=self.checker_result_col_widths,sbar_background_color="#C0BFC0",  sbar_trough_color = "#E5E5EA",
        #     font = self.checker_result_font)]
        #     ]

        self.checker_result_layout_var = {"values" : [],
            "headings" : self.checker_result_heading,
            "max_col_width":66,
            "auto_size_columns" : False,
            "display_row_numbers":False,
            "def_col_width":16,
            "justification" : "right",
            "num_rows":10,
            "key" : "-Table-",
            "row_height" : 35,
            "col_widths":self.checker_result_col_widths,"sbar_background_color":"#C0BFC0",  
            "sbar_trough_color" : "#E5E5EA",
            "font" : self.checker_result_font} 

    def setminsz_n_refresh(self, window : gui.Window, 
                           min_sz : Tuple[int, int] = (294,1)) -> gui.Window:
        
        window.set_min_size(min_sz)
        window.refresh()

        return window

    def make_issue_logger_win(self):
        """  
        Issue logger window layout
            
        returns:
        make issue logger window when called 
        """
        layout = [
            [gui.Text('Session', size=(10, 1)), gui.Text("",key= '-win2_session-')],
            [gui.Text('Please describe the issue (be precise):')],
            [gui.MLine(size=(70,15), key='-win2_issue_description-')],
            [gui.Button('Submit',key="-submit_issue-"), gui.Button('Cancel')]]
        return gui.Window('Issue Logger', layout, finalize=True)

    def make_dataset_collector_win(self, save_path : pathlib.Path):

        CAM_CTR_PARAM = PO_CONST.CamControlParam()

        button_col = [
            [gui.T("Buttons",font=self.GENERAL_FONT +  self.TITLE_FONT_SZ + self.BOLD)], 
            [gui.T("Camera LIVE", font=self.GF_CFSZ, key="cam_status", background_color="#28cd41", border_width = 1)],
            [gui.Button("Take a picture/resume camera.", key="-capture-")],
            [gui.T("AF mode: ", font=self.GF_CFSZ,), 
             gui.Combo(values=('AUTO_ONCE', 'CONT_VID', 'MACRO',"CONT_PIC", "EDOF", "AFroi"), 
                       default_value='CONT_VID', readonly=True, k='-autofocus_combo-', enable_events=True)],
            [gui.Button("AUTO-exposure", key = "-Autoexposure_enable-")],
            [gui.Button("AUTO-whitebalance", key = "-Autowhitebalance_enable-")],
            [gui.Button("getLensPos", font=self.GF_CFSZ, key = "-getLensPos-"),
             gui.B("clearTerminal", font=self.GF_CFSZ, key = "-clearTerminal-")],
            [gui.HorizontalSeparator()],
            [gui.T("Lens pos 0..255 [, far . near]: ", font=self.GF_CFSZ,), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-lens_position-")],
            [gui.Slider(range = (0, 255),expand_x=True, orientation ="h",key="-AF_slider-",
                         default_value=CAM_CTR_PARAM.lensPos, enable_events=True, disable_number_display = True)],
            [gui.T(f"Exposure time [i o]: [{CAM_CTR_PARAM.expTime}]", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-exp_time-")],
            [gui.T(f"ISO [k l]: [{CAM_CTR_PARAM.sensIso}]", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-iso-")],
            [gui.T(f"WB temperature [n m]: [{CAM_CTR_PARAM.wbManual} K]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-WB_temp-")],
            [gui.HorizontalSeparator()],
            [gui.T("Press index on keyboard, +/- to adjust:", font=self.GF_CFSZ)],
            [gui.T("control : ", font=self.GF_CFSZ),
             gui.T({CAM_CTR_PARAM.control},font=self.GF_CFSZ, key="-control-")],
            [gui.T(f"1. AWB lock [{CAM_CTR_PARAM.awb_lock}]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-AWB_lock-")],
            [gui.T(f"2. AE lock [{CAM_CTR_PARAM.ae_lock}]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-AE_lock-")],
            [gui.T("3. AWB_mode: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ, key="-AWB_mode-")],
            [gui.T(f"4. AE comp [{CAM_CTR_PARAM.ae_comp}]: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ,
                                                                                            key="-AE_comp-")],
            [gui.T("5. Anti-banding: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ, key="-Anti_banding-")],
            [gui.T("6. Effect_mode: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ, key="-effect_mode-")],
            [gui.T(f"7. Brightness [{CAM_CTR_PARAM.brightness}]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-brightness-")],
            [gui.T(f"8. Contrast [{CAM_CTR_PARAM.contrast}]: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ,
                                                                                              key="-contrast-")],
            [gui.T(f"9. Saturation [{CAM_CTR_PARAM.saturation}]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-saturation-")],
            [gui.T(f"0. Sharpness [{CAM_CTR_PARAM.sharpness}]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-sharpness-")],
            [gui.T(f"[. Luma_denoise [{CAM_CTR_PARAM.luma_denoise}]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-luma_denoise-")],
            [gui.T(f"]. Chroma_denoise [{CAM_CTR_PARAM.chroma_denoise}]: ", font=self.GF_CFSZ), 
             gui.T("AUTO",font=self.GF_CFSZ, key="-chroma_denoise-")]
            ]
        
        camera_graph = [
            [gui.Text('Image/Camera\'s view', font=self.GENERAL_FONT +  self.TITLE_FONT_SZ + self.BOLD)],
            [gui.Image(filename="", key='-GRAPH-',background_color="white")]
            ]

        txt_output = [
            [gui.Text('Terminal outputs', font=self.GENERAL_FONT +  self.TITLE_FONT_SZ + self.BOLD)],
            [gui.MLine(size=(30,5), key='-OUTPUT-'+gui.WRITE_ONLY_KEY, expand_x=True, 
                       reroute_stdout=True,reroute_stderr=True)]
            ]

        layout = [
            [gui.Text(f"Images are saved in \"current_date\"\\\"dataset_raw\" folder in: {save_path}")],
            [gui.vtop(gui.Column(button_col, element_justification="l")),
             gui.vtop(gui.Column(camera_graph, element_justification="c"))],
            [gui.Column(txt_output, element_justification="l", expand_x=True)]
            ]
        
        return gui.Window("Dataset image taker",layout,grab_anywhere=True,
                            resizable=True,finalize=True,return_keyboard_events=True)

    def make_main_win(self, AUTO_SAVE_IMG = SES_SETTING["session"]["AUTO_SAVE_IMG"], 
                      ENABLE_TEST_BUTTON = SES_SETTING["session"]["ENABLE_TEST_BUTTON"], 
                      DEV_MODE = SES_SETTING["session"]["DEV_MODE"]):
        
        checker_result_layout = [
            [gui.Table(**self.checker_result_layout_var)]
            ]

        button_col = [
            [gui.T("Buttons",font=self.GENERAL_FONT +  self.TITLE_FONT_SZ + self.BOLD)], 
            [gui.T("Camera LIVE", font=self.GF_CFSZ, key="camera_status", background_color="#28cd41", border_width = 1)],
            [gui.Button("Preview obj det", key="-preview_obj_det-")],
            [gui.Button("Start inspection", key="-start_inspection-")],
            [gui.Button("Take a picture/resume camera.", key="-capture-", disabled=True)],
            [gui.Button("Doc", key="-document-", disabled=True)],
            [gui.Button("Products/labels", key="-product_label-", disabled=True)], 
            [gui.Button("CoC vs PTD vs Serial tag", key="-coc_vs_ptd_vs_st-", disabled=True)],
            [gui.Button("Inspection results", key="pandas", disabled=True)],
            [gui.Button("Submit results", key="-submit_result-", disabled=True)],
            [gui.Button("Reset", key="reset", disabled=True)],
            [gui.Button("Report issue", key="-open_issue_win-", disabled=True)],
            [gui.HorizontalSeparator()],
            [gui.T("AF mode: ", font=self.GF_CFSZ), gui.Combo(values=tuple(PO_CONST.CAM_AF_OPT.keys()), default_value='CONT_VID', readonly=True, k='-autofocus_combo-', enable_events=True , size = 25)],
            [gui.T("Lens pos 0..255 [, far . near]: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ, key="-lens_position-")],
            [gui.HorizontalSeparator()],
            [gui.T("Press index on keyboard, +/- to adjust:", font=self.GF_CFSZ)],
            [gui.T(f"4. AE comp [{PO_CONST.CAM_CTR_PARAM['ae_comp']}]: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ, key="-AE_comp-")],
            [gui.T(f"7. Brightness [{PO_CONST.CAM_CTR_PARAM['brightness']}]: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ, key="-brightness-")],
            ]

        if not AUTO_SAVE_IMG:
            button_col.append([gui.Button("Save session img", key="-save_session_img-",  )])
            
        if ENABLE_TEST_BUTTON:
            button_col.append([gui.Button("custom test btn", key="-custom_test_button-",  )])
        
        # camera_graph = [
        #     [gui.Text('Image/Camera\'s view',font=GENERAL_FONT +  TITLE_FONT_SZ + BOLD)],
        #     [gui.Image(filename="", key='-GRAPH-',bg_color="white")]
        #     ]

        txt_output = [
            [gui.Text('Terminal outputs',font=self.GENERAL_FONT +  self.TITLE_FONT_SZ + self.BOLD),
            gui.Checkbox("See terminal history", default=False, key = "-terminal_history-", enable_events=True)],
            [gui.pin(gui.MLine(size=(70,29), key='-OUTPUT-'+gui.WRITE_ONLY_KEY, 
            # sbar_bg_color="#C0BFC0",  sbar_trough_color = "#E5E5EA",
            font = self.TERMINAL_FONT+self.CAPTION_FONT_SZ,expand_x=True, reroute_stdout=True,reroute_stderr=True,visible=False))],
            [gui.pin(gui.MLine(size=(70,29), key='-OUTPUT2-'+gui.WRITE_ONLY_KEY, 
            # sbar_bg_color="#C0BFC0",  sbar_trough_color = "#E5E5EA",
            font = self.TERMINAL_FONT+self.CAPTION_FONT_SZ,expand_x=True, reroute_stdout=True,reroute_stderr=True, visible=True))]
            ]

        if DEV_MODE:
            txt_output.append([gui.InputText(key="-simulation_index-"), gui.Button("Change simulation idx", key = "-update_sim_idx-")])
            # txt_output.append([gui.T(str(top_path.parts[-4:]),key="-current_path-")])
            txt_output.append([gui.T("",key="-current_path-")])
            txt_output.append([gui.InputText(key="-image_path-"), gui.Button("Change img folder path", key = "-update_top_path-")])
    
        w, h = gui.Window.get_screen_size()
        if w>h:
            layout = [
            [gui.vtop(gui.Column(button_col, element_justification="l")),
            # gui.vtop(gui.Column(camera_graph, element_justification="c")),
            gui.vtop(gui.Column(txt_output, element_justification="l",expand_x=True),expand_x=True)],
            checker_result_layout
            ]
        else:
            layout = [
            [gui.vtop(gui.Column(button_col, element_justification="l")),
            # gui.vtop(gui.Column(camera_graph, element_justification="c"))
            ],
            [gui.Column(txt_output, element_justification="l", expand_x=True)],
            checker_result_layout
            ]
        return gui.Window("Polaris", layout, grab_anywhere=True, resizable=True, finalize=True, return_keyboard_events=True)

    def make_cobot_GUI(self, lensPos = 30,  # this is a random number
                       DEV_MODE = SES_SETTING["session"]["DEV_MODE"]):   

        checker_result_layout = [
            [gui.Table(**self.checker_result_layout_var)]
            ]
        
        layout = [
            [gui.T("Buttons",font=self.GENERAL_FONT +  self.TITLE_FONT_SZ + self.BOLD)], 
            [gui.T("Camera LIVE", font=self.GENERAL_FONT +  self.CAPTION_FONT_SZ, key="camera_status", background_color="#28cd41", border_width = 1)],
            [gui.Button("Preview obj det", key="-preview_obj_det-")],
            [gui.Button("Start inspection", key="-start_inspection-")],
            [gui.Button("Take a picture/resume camera.", key="-capture-", disabled=True)],
            [gui.Button("Doc", key="-document-", disabled=False)],
            [gui.Button("Products/labels", key="-product_label-")],  
            [gui.Button("CoC vs PTD vs Serial tag", key="-coc_vs_ptd_vs_st-")],
            [gui.Button("Inspection results", key="pandas")],
            # [gui.Button("Submit results", key="-submit_result-", disabled=True)],
            [gui.Button("Reset", key="reset", disabled=True)],
            # [gui.Button("Report issue", key="-open_issue_win-", disabled=True)],
            [gui.HorizontalSeparator()],
            [gui.T("AF mode: ", font=self.GF_CFSZ), gui.Combo(values=tuple(PO_CONST.CAM_AF_OPT.keys()), default_value='CONT_VID', readonly=True, k='-autofocus_combo-', enable_events=True , size = 25)],
            [gui.T("Lens pos 0..255 [, far . near]: ", font=self.GF_CFSZ), gui.T("AUTO",font=self.GF_CFSZ, key="-lens_position-")],
            [gui.Slider(range = (0, 255), expand_x=True, orientation ="h",key="-AF_slider-", default_value=lensPos, enable_events=True, disable_number_display = True)],
            [gui.T("laser"), gui.Slider(range = (0, 1200), expand_x=True, orientation ="h",key="-laser_slider-", default_value=0, enable_events=True)],
            # [gui.Button("take_pict",key="take_pict"),gui.Button("take_pictv2",key="take_pictv2"),gui.Button("take_pictv3",key="take_pictv3")],
            [gui.HorizontalSeparator()],
            [gui.Button("movej_home",key="movej_home"),gui.Button("movej_top",key="movej_top"),gui.Button("movej_mid300",key="movej_mid300"),gui.Button("movej_frontChassis",key="movej_frontChassis")],
            [gui.Button("movel_tcp", key="movel_tcp")],
            [gui.HorizontalSeparator()],
            # [gui.Button("take_pict_livelens",key="take_pict_livelens"), gui.Button("take_pict_livelens2",key="take_pict_livelens2")],
            # [gui.Button("take_pict_livelens3",key="take_pict_livelens3"),gui.Button("take_pict_livelens4",key="take_pict_livelens4")], 
            [gui.Button("demo0",key="demo0"), gui.Button("demo1",key="demo1"), gui.Button("demo2",key="demo2")],
            [gui.Button("demo_w_extract",key="demo_w_extract")],
            # [gui.Button("multiple_adjust1",key="multiple_adjust1"),gui.Button("multiple_adjust2",key="multiple_adjust2")],
            [gui.Button("tester",key="tester"),gui.Button("-test_ocr-",key="-test_ocr-"), gui.B("-test_objdet-", k="-test_objdet-")],
            [gui.Button("freedrive: OFF",key="-freedrive_mode-")],
            [gui.Button("-close_cobot_process-",key="-close_cobot_process-"),gui.Button("-restart_cobot_process-",key="-restart_cobot_process-")],
            [gui.Button('-CVdestroyAllWindows-', key='-CVdestroyAllWindows-')]]
        
        txt_output = [
            [gui.Text('Terminal outputs',font=self.GENERAL_FONT +  self.TITLE_FONT_SZ + self.BOLD),
            gui.Checkbox("See terminal history", default=False, key = "-terminal_history-", enable_events=True)],
            [gui.pin(gui.MLine(size=(70,15), key='-OUTPUT-'+gui.WRITE_ONLY_KEY, 
            # sbar_bg_color="#C0BFC0",  sbar_trough_color = "#E5E5EA",
            font = self.TERMINAL_FONT+self.CAPTION_FONT_SZ,expand_x=True, reroute_stdout=True,reroute_stderr=True,visible=False))],
            [gui.pin(gui.MLine(size=(70,15), key='-OUTPUT2-'+gui.WRITE_ONLY_KEY, 
            # sbar_bg_color="#C0BFC0",  sbar_trough_color = "#E5E5EA",
            font = self.TERMINAL_FONT+self.CAPTION_FONT_SZ,expand_x=True, reroute_stdout=True,reroute_stderr=True, visible=True))]
            ]
        
        if DEV_MODE:
            txt_output.append([gui.InputText(key="-simulation_index-"), gui.Button("Change simulation idx", key = "-update_sim_idx-")])
            # txt_output.append([gui.T(str(top_path.parts[-4:]),key="-current_path-")])
            txt_output.append([gui.T("",key="-current_path-")])
            txt_output.append([gui.InputText(key="-image_path-"), gui.Button("Change img folder path", key = "-update_top_path-")])

        layout = [
            [gui.vtop(gui.Column(layout, element_justification="l")),
            # gui.vtop(gui.Column(camera_graph, element_justification="c"))
            ],
            [gui.Column(txt_output, element_justification="l", expand_x=True)] 
            # checker_result_layout
            ]

        return gui.Window("Polaris (Cobot)", layout, grab_anywhere=True, resizable=True, finalize=True, return_keyboard_events=True)

    def ask_productI_face(self) -> Optional[str]:  # @ last_update 01 Jun 2023

        """
        asks user which face of the product to inspect
        """
        product_faces = ["top","front","back","left","right"]

        max_length = max(len(text) for text in product_faces)

        buttons = {key: gui.Button(key, size=(max_length, 1), key=key)  for key in product_faces}

        buttons["Cancel"] = gui.Cancel(size=(max_length*2,1)) 

        # Arrange the buttons in custom placement
        layout = [
                [gui.Column([[buttons['top']],], justification='center')],
                [gui.Column([[buttons['front'],buttons['back']]], justification='center')],
                [gui.Column([[buttons['left'],buttons['right']]], justification='center')],
                [gui.Column([[buttons['Cancel']]], justification='center')]
            ]
                
        window = gui.Window('', layout, finalize=True,grab_anywhere=True,disable_close=True)

        event, _ = window.read(close=True)

        if event == "Cancel":
            return None
        else:
            return event 

    def ask_is_fresh_unit(self):
        layout  = [[gui.Column([[gui.T("Is this a fresh unit?", justification="center")],
                                [gui.T("Fresh units: select YES\n\nOtherwise: select NO. Please perform manual inspection.",justification="left")], 
                                [gui.Yes(s=10), gui.No(s=10)]],element_justification='centre', expand_x=True, key="temp_col")]]

        window = gui.Window("Polaris", layout, finalize=True,grab_anywhere=True,disable_close=True)

        window_w, window_h = window["temp_col"].get_size()
        window.set_min_size((294,window_h))
        window.refresh()
        user_fresh_unit_choice, _ = window.read(close=True)

        logger.info(f"user_fresh_unit_choice : {user_fresh_unit_choice}")

        return user_fresh_unit_choice

    def ask_is_Axie_involved(self):
        window = gui.Window("Polaris", [[gui.Column(
                    [[gui.T('Does the inspection involves product in an AXIE chassis?',justification="center")], 
                    [gui.Yes(s=10), gui.No(s=10)]],element_justification='centre', expand_x=True, key="temp_col")]], 
                        disable_close=True, finalize=True)
        
        window_w, window_h = window["temp_col"].get_size()
        window.set_min_size((294,window_h))
        window.refresh()
        bundle_present, _ = window.read(close=True)

        logger.info(f"user_bundle_present_choice : {bundle_present}")

        return bundle_present

    def proceed_doc_mode(self) -> bool:
        BTN_PER_ROW = 2
        EXT = "No"
        X = ["Yes", EXT]

        layout = []
        
        txts = ["Inspect document?\n", "Yes: Inspect doc in image\nNo: Image will not be analysed and ignored", ]
        for t in txts:
            layout.append([gui.T(t)])
                                
        max_length: int = max(len(text) for text in X)
        buttons = [gui.Button(key, size = (max(max_length, 15), 1), key=key) for key in X]

        new_rows = [[gui.Column([buttons[i:i+BTN_PER_ROW]], justification='center')] \
                    for i in range(0, len(buttons), BTN_PER_ROW)]
        layout.extend(new_rows)

        window = gui.Window(POLRS, layout, finalize=True, 
                            grab_anywhere=True, disable_close=True)
        window.set_min_size((294, 1))
        window.refresh()

        event, _ = window.read(close=True)

        logger.info(f"Inspect doc? User selected: {event}")

        if event == EXT:
            return False
        else:
            return True

    def ask_xyz_for_movel_tcp(self):
        layout = [[gui.Text('Enter values for x, y, and z [cm]:')],
                [gui.Text('x:',), gui.InputText(k="x")],
                [gui.Text('y:',), gui.InputText(k="y")],
                [gui.Text('z:',), gui.InputText(k='z')],
                [gui.Button('Ok', k= "ok"), gui.Button('Cancel')]
                ]

        window = gui.Window('Input values for x, y, and z', layout, finalize=True,grab_anywhere=True,disable_close=True)

        event, values = window.read(close=True)

        p_to = {"x":0,"y":0,"z":0}
        if event == "ok":
            for k in p_to:
                if len(values[k])>0:
                    p_to[k] = float(values[k])/100 # unit change to [m]
                else:
                    p_to[k] = 0
        
        print(f"p_to: {p_to}")

        return p_to

    def ask_subjects_for_closeup(self) -> List[str]:
        KEY = "-all-"
        KEY2 = "-labels_only-"
        
        X = PO_CONST.CLOSEUP_SUBJECTS
        max_length : int = max(len(text) for text in X)
        
        buttons = [gui.Button(key, size=(max_length, 1), key=key) for key in X]
        
        all_subjects : str = ', '.join(X)
        buttons.append(gui.Button(all_subjects, size=(len(all_subjects), 1), key=KEY))
        
        layout = [[gui.Column([[gui.T("Select subjects for closeup")]], 
                              justification = "center")]]
        
        new_rows = [[gui.Column([buttons[i:i+2]], justification='center')] \
                    for i in range(0, len(buttons), 2)]
        
        layout.extend(new_rows)
        
        layout.append([gui.Column([[gui.Button(KEY2, size=(len(all_subjects), 1),
                                               key=KEY2)]], justification = "center")])
        
        window = gui.Window(POLRS, layout, finalize=True, disable_close=True)
        window.set_min_size((294, 1))
        window.refresh()
        
        subject, _ = window.read(close=True)
        
        if subject == KEY:
            return X
        elif subject == KEY2:
            return PO_CONST.CLOSEUP_LABELS
        else:
            return [subject]

    def ask_product_type_for_frontI(self) -> str:
        BTN_PER_ROW = 1
        X = {"prod_0_front": 'module (No Chassis)', 
             "prod_1_front" : "pod"}

        layout = [[gui.T("Select product type:")]]

        max_length: int = max(len(text) for text in X.values())
        buttons = [gui.Button(btn_str, size = (max(max_length, 15), 1), 
                              key=key) for key,btn_str in X.items()]
        new_rows = [[gui.Column([buttons[i:i+BTN_PER_ROW]], justification='center')] \
                    for i in range(0, len(buttons), BTN_PER_ROW)]
        layout.extend(new_rows)

        window = gui.Window(POLRS, layout, finalize=True, 
                            grab_anywhere=True, disable_close=True)
        
        window = self.setminsz_n_refresh(window)

        product_type, _ = window.read(close=True)
        logger.info(f"User selected product_type \"{product_type}\" for front inspection.")
        
        return product_type

    def ask_xy_for_AFonROI(self):
        layout = [[gui.Text('Enter values for xmin, ymin, xmax, ymax')],
                [gui.Text('xmin:',), gui.InputText(k="xmin")],
                [gui.Text('ymin:',), gui.InputText(k="ymin")],
                [gui.Text('xmax:',), gui.InputText(k='xmax')],
                [gui.Text('ymax:',), gui.InputText(k='ymax')],
                [gui.Button('Ok', k= "ok"), gui.Button('Cancel')]
                ]

        window = gui.Window('Input values for x, y, and z', layout, finalize=True,grab_anywhere=True,disable_close=True)

        event, values = window.read(close=True)
        
        AFRoi = {"xmin":0,"ymin":0,"xmax":0,"ymax":0}
        if event == "ok":
            for k in AFRoi:
                if len(values[k])>=1:
                    AFRoi[k] = int(values[k])
                else:
                    AFRoi[k] = 0

        return po_utils.bbox_XY(xmin = AFRoi["xmin"], ymin = AFRoi["ymin"], 
                                xmax = AFRoi["xmax"], ymax = AFRoi["ymax"])
    
    def ask_proceed_seekSN(self, issues: str, title : str, mn_sn: str) -> bool:
        message = f"The following issues have been discovered for {mn_sn}:\n\n"\
              + issues + f"\n\nDo you wish to proceed finding labels with the same\
                  {PO_CONST.CONSTANT_STR.SN}?"
        message += "\n\nYes: Checker will only seek using labels/docs found\
              without issues.\nNo : Checker will mark this as fail."
        proceed, _ =  gui.Window(title, [[gui.T(f'{message}')], 
                                      [gui.Yes(s=10), gui.No(s=10)]], 
                                      disable_close=True).read(close=True)
        
        proceed = bool(strtobool(proceed))
        
        logger.info(f"User allow proceed for SN seeker: {proceed}")
        
        return proceed

    def ask_proceed_MNSNOPT_Checker(self, issues: str, title : str, mn_sn: str) -> bool:
        message = f"The following issues have been discovered for {mn_sn}:\n\n"\
              + issues +"\n\nDo you wish to proceed checking MN SN OPT?"
        message += "\n\nYes: Checker will only check using labels/docs found\
              without issues.\nNo : Checker will mark this as fail."
        proceed, _ =  gui.Window(title, [[gui.T(f'{message}')], 
                                      [gui.Yes(s=10), gui.No(s=10)]], 
                                      disable_close=True).read(close=True)
        
        proceed = bool(strtobool(proceed))
        
        logger.info(f"User allow proceed for MN_SN_OPT check: {proceed}")
        
        return proceed
    
    def prompt_user_check_doc(self, doc_name : str, modelno_sn = None, space_string = str(' '*5)):
        DOC = PO_CONST.DocsRequiringInspection()
        X = ["Pass", "Fail"]
        
        layout = [[gui.T(modelno_sn)]]

        user_respond_to_doc_check = False
        if doc_name == DOC.M8070B_doc:
            layout.extend([[gui.Text('\nPlease check the firmware document for M8070B. Click:')]])

        elif doc_name == DOC.coe:
            layout.extend([[gui.Text('\nPlease check: CoE (License of Redemption Entitlement Certificate) document(s).\n\nClick:')]])

        new_rows = [[gui.Text(f'{space_string}Pass: If the document is present & contents are correct/doc is not required.')],
                      [gui.Text(f'{space_string}Fail: If document is missing/ has issues.')]]
                    #   [gui.Text(f'{space_string}Not Required: {doc_name} is not required')]]

        layout.extend(new_rows)

        max_length: int = max(len(text) for text in X)
        buttons = [gui.Button(key, size = (max(max_length, 15), 1), key=key) for key in X]
        new_rows = [[gui.Column([buttons[i:i+2]], justification='center')] \
                    for i in range(0, len(buttons), 2)]
        layout.extend(new_rows)

        user_respond_to_doc_check, _  =  gui.Window(POLRS, layout, resizable=True, finalize=True, 
                                                    grab_anywhere=True, modal=True, disable_close=True).read(close=True)
        
        if user_respond_to_doc_check == X[0]:
            return True
        else:
            return False


