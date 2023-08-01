import logging
import numpy as np
import sys
import time
import threading
from flask import Flask, request

import utils.util
from utils.CentralStore import IPCount
from utils.DatasetStore import LocalDataset
from utils.EnvStore import EnvStore
from utils.ModelStore import CentralModelStore
from utils.Trainer import Trainer
from utils.util import ColoredLogger
from models.Fed import fed_avg

logging.setLoggerClass(ColoredLogger)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)

env_store = EnvStore()
local_dataset = LocalDataset()
central_model_store = CentralModelStore()
ipCount = IPCount()
trainer_pool = {}  # multiple thread trainers stored in this map


def init_trainer():
    trainer = Trainer()
    trainer.uuid = fetch_uuid()

    load_result = trainer.init_model(env_store.args.model, env_store.args.dataset, env_store.args.device,
                                     local_dataset.image_shape)
    if not load_result:
        sys.exit()

    # trained the initial local model, which will be treated as first global model.
    trainer.model_store.local_model.train()
    # generate md5 hash from model, which is treated as global model of previous round.
    w = trainer.model_store.local_model.state_dict()
    central_model_store.update_global_model(w, epochs=-1)  # -1 means the initial global model
    trainer.model_store.update_global_model(w)
    trainer_pool[trainer.uuid] = trainer
    return trainer.uuid


def train(trainer_uuid):
    trainer = trainer_pool[trainer_uuid]
    logger.debug("Train local model for user: {}, epoch: {}.".format(trainer.uuid, trainer.epoch))

    trainer.round_start_time = time.time()
    # calculate initial model accuracy, record it as the benchmark.
    if trainer.is_first_epoch():
        if (not env_store.args.dis_acc_test and trainer.uuid == 1) or env_store.args.dis_acc_test:
            trainer.init_time = time.time()
            # download initial global model
            body_data = {
                "message": "global_model",
                "epochs": -1,
            }
            detail = utils.util.post_msg_trigger(env_store.trigger_url, body_data)
            global_model_compressed = detail.get("global_model")
            w_glob = utils.util.decompress_tensor(global_model_compressed)
            trainer.load_model(w_glob)
            trainer.evaluate_model_with_log(local_dataset, env_store.args, record_epoch=0, clean=True)
    else:
        trainer.load_model(trainer.model_store.global_model)

    train_start_time = time.time()
    w_local, w_loss = trainer.train(local_dataset, env_store.args)
    trainer.round_train_duration = time.time() - train_start_time

    # evaluate the accuracy of the trained model
    acc_local, loss_local = trainer.evaluate_model(local_dataset, env_store.args)
    # send local model to the first node
    w_local_compressed = utils.util.compress_tensor(w_local)
    body_data = {
        "message": "upload_local_w",
        "w_compressed": w_local_compressed,
        "acc_local": acc_local,
        "loss_local": loss_local,
        "uuid": trainer.uuid,
        "from_ip": env_store.from_ip,
    }
    utils.util.post_msg_trigger(env_store.trigger_url, body_data)


def gather_local_w(local_uuid, from_ip, w_compressed, acc_local, loss_local):
    ipCount.set_map(local_uuid, from_ip)
    if central_model_store.local_models_add_count(local_uuid, utils.util.decompress_tensor(w_compressed),
                                                  acc_local, loss_local, env_store.args.num_users):
        logger.debug("Gathered enough w, randomly pick up local models.")
        model_select_num = max(int(env_store.args.frac * env_store.args.num_users), 1)
        if env_store.args.scheme == "datasize":
            dataset_sizes = [len(train_dataset) for _, train_dataset in local_dataset.dict_users.items()]
            user_ids, weights = select_models_on_datasize(dataset_sizes, model_select_num)
        elif env_store.args.scheme == "gradiv_max":
            user_ids, weights = select_models_on_divergence(central_model_store.local_models,
                                                            central_model_store.global_model, model_select_num, True)
        elif env_store.args.scheme == "gradiv_min":
            user_ids, weights = select_models_on_divergence(central_model_store.local_models,
                                                            central_model_store.global_model, model_select_num, False)
        elif env_store.args.scheme == "entropy_max":
            user_ids, weights = select_models_on_entropy(central_model_store.local_models,
                                                         central_model_store.global_model, model_select_num, True)
        elif env_store.args.scheme == "entropy_min":
            user_ids, weights = select_models_on_entropy(central_model_store.local_models,
                                                         central_model_store.global_model, model_select_num, False)
        elif env_store.args.scheme == "accuracy":
            user_ids, weights = select_models_on_acc(central_model_store.local_models_acc, model_select_num)
        elif env_store.args.scheme == "loss_max":
            user_ids, weights = select_models_on_loss(central_model_store.local_models_loss, model_select_num, True)
        elif env_store.args.scheme == "loss_min":
            user_ids, weights = select_models_on_loss(central_model_store.local_models_loss, model_select_num, False)
        else:
            user_ids, weights = select_models_on_random(model_select_num)
        w_glob = fed_avg(central_model_store.local_models, central_model_store.global_model, env_store.args.device,
                         user_ids, weights)
        # reset local models after aggregation
        central_model_store.local_models_reset()
        # save global model
        central_model_store.update_global_model(w_glob)
        for uuid in ipCount.get_keys():
            body_data = {
                "message": "release_global_w",
                "w_compressed": central_model_store.global_model_compressed,
                "uuid": uuid,
            }
            user_url = "http://" + ipCount.get_map(uuid) + ":" + str(env_store.args.fl_listen_port) + "/trigger"
            utils.util.http_client_post(user_url, body_data)


def select_models_on_datasize(dataset_sizes, model_select_num):
    dataset_size_dict = {}
    for local_uuid in range(1, env_store.args.num_users + 1):
        dataset_size_dict[local_uuid] = dataset_sizes[local_uuid - 1]
    logger.debug("local users' dataset sizes: {}".format(dataset_size_dict))
    sorted_keys = sorted(dataset_size_dict.keys(), key=dataset_size_dict.get, reverse=True)
    logger.debug("sorted local uuid by local dataset sizes: {}".format(sorted_keys))
    selected_uuids = sorted_keys[:model_select_num]
    weights = normalize([dataset_size_dict[uuid] for uuid in selected_uuids])
    return selected_uuids, weights


# max_divergence: if true, return uuids with bigger divergences; otherwise, return uuids with smaller divergences.
def select_models_on_divergence(local_models, glob_model, model_select_num, max_divergence):
    divergence_dict = {}
    for local_uuid in range(1, env_store.args.num_users + 1):
        divergence_dict[local_uuid] = utils.util.calculate_divergence(local_models[local_uuid], glob_model,
                                                                      env_store.args.device)
    logger.debug("local models' divergence: {}".format(divergence_dict))
    sorted_keys = sorted(divergence_dict.keys(), key=divergence_dict.get, reverse=max_divergence)
    logger.debug("sorted local uuid by local models' divergence: {}".format(sorted_keys))
    selected_uuids = sorted_keys[:model_select_num]
    if max_divergence:
        weights = normalize([divergence_dict[uuid] for uuid in selected_uuids])
    else:
        # if select the minimal ones, assign weights by the reciprocals
        weights = normalize([1 / divergence_dict[uuid] for uuid in selected_uuids])
    return selected_uuids, weights


def select_models_on_entropy(local_models, glob_model, model_select_num, max_entropy):
    entropy_dict = {}
    for local_uuid in range(1, env_store.args.num_users + 1):
        entropy_dict[local_uuid] = utils.util.calculate_entropy(local_models[local_uuid], glob_model,
                                                                env_store.args.device)
    logger.debug("local models' entropy: {}".format(entropy_dict))
    sorted_keys = sorted(entropy_dict.keys(), key=entropy_dict.get, reverse=max_entropy)
    logger.debug("sorted local uuid by local models' entropy: {}".format(sorted_keys))
    selected_uuids = sorted_keys[:model_select_num]
    if max_entropy:
        weights = normalize([entropy_dict[uuid] for uuid in selected_uuids])
    else:
        # if select the minimal ones, assign weights by the reciprocals
        weights = normalize([1 / entropy_dict[uuid] for uuid in selected_uuids])
    return selected_uuids, weights


def select_models_on_acc(local_acc_dict, model_select_num):
    logger.debug("local models' accuracies: {}".format(local_acc_dict))
    sorted_keys = sorted(local_acc_dict.keys(), key=local_acc_dict.get, reverse=True)
    logger.debug("sorted local uuid by local models' accuracies: {}".format(sorted_keys))
    selected_uuids = sorted_keys[:model_select_num]
    weights = normalize([local_acc_dict[uuid] for uuid in selected_uuids])
    return selected_uuids, weights


def select_models_on_loss(local_loss_dict, model_select_num, max_loss):
    logger.debug("local models' losses: {}".format(local_loss_dict))
    sorted_keys = sorted(local_loss_dict.keys(), key=local_loss_dict.get, reverse=max_loss)
    logger.debug("sorted local uuid by local models' losses: {}".format(sorted_keys))
    selected_uuids = sorted_keys[:model_select_num]
    if max_loss:
        weights = normalize([local_loss_dict[uuid] for uuid in selected_uuids])
    else:
        weights = normalize([1 / local_loss_dict[uuid] for uuid in selected_uuids])
    return selected_uuids, weights


def select_models_on_random(model_select_num):
    selected_uuids = np.random.choice(range(1, env_store.args.num_users + 1), size=model_select_num, replace=False)
    weights = normalize([1] * model_select_num)
    return selected_uuids, weights


def normalize(weights_array):
    logger.debug("Weights to be normalized: {}".format(weights_array))
    normalized_weights = [num / sum(weights_array) for num in weights_array]
    logger.debug("Normalized weights: {}".format(normalized_weights))
    return normalized_weights


def receive_global_w(trainer_uuid, w_glob_compressed):
    trainer = trainer_pool[trainer_uuid]
    logger.debug("Received latest global model for user: {}, epoch: {}.".format(trainer.uuid, trainer.epoch))

    # load hash of new global model, which is downloaded from the leader
    w_glob = utils.util.decompress_tensor(w_glob_compressed)
    trainer.model_store.update_global_model(w_glob)

    # finally, evaluate the global model
    trainer.load_model(w_glob)
    if (not env_store.args.dis_acc_test and trainer.uuid == 1) or env_store.args.dis_acc_test:
        trainer.evaluate_model_with_log(local_dataset, env_store.args, record_communication_time=True)

    trainer.epoch += 1
    if trainer.epoch <= env_store.args.epochs:
        logger.info("########## EPOCH #{} ##########".format(trainer.epoch))
        train(trainer.uuid)
    else:
        # save the latest global model before exiting for further evaluation
        logger.debug("Save the latest global model")
        if trainer.uuid == 1:
            trainer.save_model()
        logger.info("########## ALL DONE! ##########")
        body_data = {
            "message": "shutdown_python",
            "uuid": trainer.uuid,
            "from_ip": env_store.from_ip,
        }
        utils.util.post_msg_trigger(env_store.trigger_url, body_data)


def fetch_uuid():
    body_data = {
        "message": "fetch_uuid",
    }
    detail = utils.util.post_msg_trigger(env_store.trigger_url, body_data)
    uuid = detail.get("uuid")
    return uuid


def load_uuid():
    new_id = ipCount.get_new_id()
    detail = {"uuid": new_id}
    return detail


def load_global_model(epochs):
    if epochs == central_model_store.global_model_version:
        detail = {
            "global_model": central_model_store.global_model_compressed,
        }
    else:
        detail = {
            "global_model": None,
        }
    return detail


def start_train():
    time.sleep(env_store.args.start_sleep)
    trainer_uuid = init_trainer()
    train(trainer_uuid)


def flask_route(app):
    @app.route("/trigger", methods=["GET", "POST"])
    def trigger_handler():
        # For POST
        if request.method == "POST":
            data = request.get_json()
            status = "yes"
            detail = {}
            message = data.get("message")
            if message == "fetch_uuid":
                detail = load_uuid()
            elif message == "global_model":
                detail = load_global_model(data.get("epochs"))
            elif message == "upload_local_w":
                threading.Thread(target=gather_local_w, args=(
                    data.get("uuid"), data.get("from_ip"), data.get("w_compressed"),
                    data.get("acc_local"), data.get("loss_local"))).start()
            elif message == "release_global_w":
                threading.Thread(target=receive_global_w, args=(data.get("uuid"), data.get("w_compressed"))).start()
            elif message == "shutdown_python":
                threading.Thread(target=utils.util.shutdown_count, args=(
                    data.get("uuid"), data.get("from_ip"), env_store.args.fl_listen_port,
                    env_store.args.num_users)).start()
            elif message == "shutdown":
                threading.Thread(target=utils.util.exit_process, args=(env_store.args.exit_sleep,)).start()
            response = {"status": status, "detail": detail}
            return response


def main():
    # init environment arguments
    env_store.init()
    # init local dataset
    local_dataset.init_local_dataset(env_store.args.dataset, env_store.args.dataset_size, env_store.args.num_users,
                                     env_store.args.dis_acc_test, env_store.args.noniid_zipf_s,
                                     env_store.args.dataset_size_variant)
    # set logger level
    logger.setLevel(env_store.args.log_level)

    for _ in range(env_store.args.num_users):
        logger.debug("start new thread")
        threading.Thread(target=start_train, args=()).start()

    flask_app = Flask(__name__)
    flask_route(flask_app)
    logger.info("start serving at " + str(env_store.args.fl_listen_port) + "...")
    flask_app.run(host="0.0.0.0", port=int(env_store.args.fl_listen_port))


if __name__ == "__main__":
    main()
