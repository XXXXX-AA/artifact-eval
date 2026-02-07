class MyMessage(object):
    """
        message type definition
    """
    # message to neighbor
    MSG_TYPE_INIT = 1
    MSG_TYPE_SEND_MSG_TO_NEIGHBOR = 2
    MSG_TYPE_METRICS = 3

    # message to coordinator
    MSG_TYPE_CLIENT_TO_COORDINATOR = 4
    MSG_TYPE_COORDINATOR_TO_CLIENT = 5

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_PARAMS_1 = "params1"
    MSG_ARG_KEY_SPARSE_PARAMS_1 = "sparse_params1"
    MSG_ARG_KEY_SPARSE_INDEX_1 = "sparse_index1"
    MSG_ARG_KEY_QUANT_PARAMS_1 = "quant_params1"
    MSG_ARG_KEY_SIGN_PARAMS_1  = "sign_params1"

    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_index"

    MSG_ARG_KEY_TRAIN_METRICS = "train_metrics"
    MSG_ARG_KEY_TEST_METRICS = "test_metrics"

    MSG_TYPE_CLUSTER_INFO      = 2001  # NOTE: comment translated from Chinese
    MSG_ARG_KEY_LOSS_DECREASED = "loss_decreased"
    MSG_ARG_KEY_CLUSTER_NAME    = "cluster_name"
    MSG_ARG_KEY_CLUSTER_NEIGHBORS = "cluster_neighbors"
    MSG_ARG_KEY_LAST_LAYER_PARAMS = "last_layer_params"
    MSG_ARG_KEY_LOCAL_ROUND = "local_round"

    MSG_ARG_KEY_SELECTED_SHAPES = "selected_shapes"

    MSG_ARG_KEY_PROJ = "proj"  # NOTE: comment translated from Chinese

    MSG_TYPE_COALITION_JOIN_NOTIFICATION = "coalition_join_notification"  # NOTE: comment translated from Chinese

    MSG_TYPE_COAL_INFO = "coal_info"

    MSG_ARG_KEY_COALITION_ID = "coalition_id"
    MSG_ARG_KEY_COALITION_VERSION = "coalition_version"

    
    MSG_ARG_KEY_DELTAS = "coal_deltas"  # NOTE: comment translated from Chinese

