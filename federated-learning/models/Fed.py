import logging
import torch

from utils.util import ColoredLogger
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__file__)


def fed_avg(w_dict, w_glob, device, id_list=None, weight_list=None):
    # weighted aggregation with normalized weight list
    if len(w_dict) == 0:
        return w_glob
    if id_list is None:
        id_list = w_dict.keys()
    logger.debug("Aggregation id_list: {}".format(id_list))
    w_avg = {}
    for k in w_glob.keys():
        for i in range(len(id_list)):
            if k not in w_avg:
                w_avg[k] = torch.zeros_like(w_glob[k], device=device)
            if device != torch.device('cpu'):
                w_dict[id_list[i]][k] = w_dict[id_list[i]][k].to(device)
            w_avg[k] = torch.add(w_avg[k], torch.mul(w_dict[id_list[i]][k], weight_list[i]))
        # w_avg[k] = torch.div(w_avg[k], len(w_dict))
    return w_avg

