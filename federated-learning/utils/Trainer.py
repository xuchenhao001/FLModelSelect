import logging
import time

from utils.ModelStore import PersonalModelStore, APFLPersonalModelStore
from utils.util import model_loader, ColoredLogger, test_model, train_model, record_log, reset_communication_time, \
    save_model

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__file__)


class Trainer:
    def __init__(self):
        self.model_store = PersonalModelStore()
        self.init_time = time.time()
        self.round_start_time = time.time()
        self.round_train_duration = 0
        self.round_test_duration = 0
        self.epoch = 1
        self.uuid = -1
        # for committee election
        self.committee_elect_duration = 0

    def init_model(self, model, dataset, device, image_shape):
        self.model_store.local_model = model_loader(model, dataset, device, image_shape)
        if self.model_store.local_model is None:
            logger.error('Error: unrecognized model')
            return False
        return True

    def load_model(self, w):
        self.model_store.local_model.load_state_dict(w)

    def dump_model(self):
        return self.model_store.local_model.state_dict()

    def save_model(self):
        # save the local model to a file
        save_model(self.uuid, self.epoch, self.model_store.local_model)

    def evaluate_model(self, dataset, args):
        self.model_store.local_model.eval()
        accuracy, loss = test_model(self.model_store.local_model, dataset, self.uuid - 1, args.local_test_bs,
                                    args.device)
        return accuracy, loss

    def evaluate_model_with_log(self, dataset, args, record_epoch=None, clean=False, record_communication_time=False):
        if record_epoch is None:
            record_epoch = self.epoch
        communication_duration = 0
        if record_communication_time:
            communication_duration = reset_communication_time()
            communication_duration += self.committee_elect_duration
        if communication_duration < 0.001:
            communication_duration = 0.0
        test_start_time = time.time()
        acc_local, _ = self.evaluate_model(dataset, args)
        test_duration = time.time() - test_start_time
        test_duration += self.round_test_duration
        total_duration = time.time() - self.init_time
        round_duration = time.time() - self.round_start_time
        train_duration = self.round_train_duration
        record_log(self.uuid, record_epoch,
                   [total_duration, round_duration, train_duration, test_duration, communication_duration], acc_local,
                   clean=clean)

    def is_first_epoch(self):
        return self.epoch == 1

    def train(self, dataset, args):
        w_local, loss = train_model(self.model_store.local_model, dataset, self.uuid - 1, args.local_ep, args.device,
                                    args.lr, args.momentum, args.local_bs)
        return w_local, loss


class APFLTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.hyper_para = 0
        self.model_store = APFLPersonalModelStore()
