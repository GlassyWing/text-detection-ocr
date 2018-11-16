from datetime import datetime
import os
import keras.backend as K

from dlocr.ctpn.core import CTPN
from dlocr.ctpn.lib.utils import get_session

default_ctpn_weight_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../weights/weights-ctpnlstm-init.hdf5")
default_ctpn_config_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../config/ctpn-default.json")

_config = CTPN.load_config(default_ctpn_config_path)
ctpn = CTPN(**_config, weight_path=default_ctpn_weight_path)


