import os

from dlocr.ctpn.core import CTPN
from dlocr.ctpn.lib.utils import get_session

default_ctpn_weight_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../weights/weights-ctpnlstm-init.hdf5")
default_ctpn_config_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../config/ctpn-default.json")

__ctpn = None


def get_or_create(ctpn_config_path=default_ctpn_config_path,
                  ctpn_weight_path=default_ctpn_weight_path):
    global __ctpn
    if __ctpn is None:
        config = CTPN.load_config(ctpn_config_path)
        __ctpn = CTPN(**config, weight_path=ctpn_weight_path)
    return __ctpn
