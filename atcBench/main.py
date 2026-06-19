
import sys

import hydra
from omegaconf import DictConfig
from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from src.utils.utils import get_initialize_config, set_seed_from_config, getdata, save_results, cleanMemory
from src.model.general import getClassifier


KNOWN_METHODS = [
    "autoencoder-is", "perplexity-is", "random-is", "gmm-is", "biois", "random",
    "no-is", "adaptive-perplexity", "adaptive-v2-perplexity", "adaptive-cluster-is"
]


def rewrite_combo_data_override(argv):
    """Rewrite `data=<dataset>_<method>` into a single template config plus field overrides."""
    rewritten = [argv[0]]

    for arg in argv[1:]:
        if arg.startswith("data="):
            value = arg.split("=", 1)[1]
            for method in sorted(KNOWN_METHODS, key=len, reverse=True):
                suffix = f"_{method}"
                if value.endswith(suffix):
                    dataset_name = value[: -len(suffix)]
                    rewritten.extend([
                        "data=template",
                        f"data.dataset_name={dataset_name}",
                        f"data.is_method={method}",
                    ])
                    break
            else:
                rewritten.append(arg)
            continue

        rewritten.append(arg)

    return rewritten



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(exp_dict: DictConfig):

    config = get_initialize_config(exp_dict)
    model_config = config.model
    dataset = config.data.dataset
    set_seed_from_config(config)

    for fold in range(config.data.nfolds):
    #for fold in [0]:

        X_train, y_train, X_test, y_test = getdata(config, fold)

        clf = getClassifier(model_config, dataset)

        #Training
        clf.fit(X_train, y_train)

        #Predicting
        y_pred_proba = clf.predict_proba(X_test)
        #y_pred = np.argmax(y_pred_proba, axis=-1)

        time_train = clf._time_to_train
        time_predict = clf._time_to_predict
        save_results(config, y_test, y_pred_proba, time_train, time_predict, fold)

        #Cleaning memory
        cleanMemory(clf)

    print("Done.")



if __name__ == "__main__":
    sys.argv = rewrite_combo_data_override(sys.argv)
    main()