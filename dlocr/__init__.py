from dlocr.ctpn.lib.utils import get_session
from dlocr.text_detection_app import TextDetectionApp
from dlocr.ctpn import default_ctpn_config_path, default_ctpn_weight_path
from dlocr.densenet import default_densenet_config_path, default_densenet_weight_path, default_dict_path

get_or_create = TextDetectionApp.get_or_create
