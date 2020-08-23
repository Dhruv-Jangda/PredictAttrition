from pathlib import Path
from dask import dataframe
from mxboard import SummaryWriter
from yaml import FullLoader, load
from mxnet.gluon.loss import L1Loss
from multiprocessing import cpu_count
from pre_process import PreProcess
from model_manager import ModelManager


if __name__ == '__main__':
    # Load parameters
    with open(".\\data\\config.yaml") as config_file:
        config = load(config_file, Loader=FullLoader)

    # Define locals
    datasets: dict = dict()
    batch_data: dict = dict()
    num_features: int = len(config["COLUMNS"]["CATEGORICAL"]["NUMERIC"])
    + len(config["COLUMNS"]["CATEGORICAL"]["STRING"])
    + len(config["COLUMNS"]["CONTINUOUS"]) + 1  # 1 for the "Id" column
    pre_proc: PreProcess = PreProcess(
        columns=config["COLUMNS"], num_workers=cpu_count())

    # Path checks
    if Path(config["PATH"]["PROCESSED"]["TRAIN"]).is_file() and\
            Path(config["PATH"]["PROCESSED"]["TEST"]).is_file():

        # Load processed data
        print("Pre-Processed data exists!\nLoading data...")
        datasets["train"] = pre_proc.load_data(
            path=config["PATH"]["PROCESSED"]["TRAIN"])
        datasets["test"] = pre_proc.load_data(
            path=config["PATH"]["PROCESSED"]["TEST"])
        print("Data loaded!")

    else:
        # Load raw data
        print("Pre-Processing of raw data is required!\nLoading data...")
        datasets["train"]: dict = pre_proc.load_data(
            path=config["PATH"]["RAW"]["TRAIN"])
        datasets["test"]: dict = pre_proc.load_data(
            path=config["PATH"]["RAW"]["TEST"])
        print("Data loaded!")

        # Impute nulls
        print("\nImputing nulls...")
        datasets["train"] = pre_proc.impute_nulls(data=datasets["train"])
        datasets["test"] = pre_proc.impute_nulls(data=datasets["test"])
        print("Nulls imputed!")

        # Remove outliers
        print("\nRemoving outliers...")
        datasets["train"] = pre_proc.remove_outliers(
            data=datasets["train"],
            threshold=config["HYPER_PARAMS"]["Z_THRESHOLD"])
        datasets["test"] = pre_proc.remove_outliers(
            data=datasets["test"], columns=config["COLUMNS"],
            threshold=config["HYPER_PARAMS"]["Z_THRESHOLD"])
        print("Outliers removed!")

        # Encode data
        print("\nEncoding data...")
        datasets["train"], train_encoding = pre_proc.encode_data(
            data=datasets["train"])
        datasets["test"], test_encoding = pre_proc.encode_data(
            data=datasets["test"], encoding=train_encoding, is_train=False)
        print("Data encoded!")

        # Save processed data
        dataframe.to_csv(datasets["train"].compute(
            num_workers=cpu_count()),
            filename=config["PATH"]["PROCESSED"]["TRAIN"])
        dataframe.to_csv(datasets["test"].compute(
            num_workers=cpu_count()),
            filename=config["PATH"]["PROCESSED"]["TEST"])

    # Split train data
    datasets["train"], datasets["validation"] = dataframe.DataFrame\
        .random_split(datasets["train"],
                      frac=[config["HYPER_PARAMS"]["TRAIN_SPLIT"],
                            1 - config["HYPER_PARAMS"]["TRAIN_SPLIT"]],
                      random_state=config["HYPER_PARAMS"]["SEED"],
                      shuffle=True)

    # Dump data to memory
    print("\nDumping data to memory...")
    datasets["train"] = datasets["train"].compute(num_workers=cpu_count())
    datasets["validation"] = datasets["validation"].compute(
        num_workers=cpu_count())
    datasets["test"] = datasets["test"].compute(num_workers=cpu_count())
    print("Data dumped!")

    # Define locals for MXNet
    model_mgr: ModelManager = ModelManager(
        loss_fn=L1Loss(), model_style=config["STYLE"].lower(),
        writer=SummaryWriter(logdir=config["PATH"]["LOGS"], flush_secs=1))

    # Prepare data
    batch_data["train"] = model_mgr.get_batch_data(
        data=datasets["train"], workers=cpu_count(), cols=config["COLUMNS"],
        batch_size=config["HYPER_PARAMS"]["BATCH_SIZE"], label="train")
    batch_data["validation"] = model_mgr.get_batch_data(
        data=datasets["validation"], workers=cpu_count(),
        batch_size=config["HYPER_PARAMS"]["BATCH_SIZE"],
        cols=config["COLUMNS"], label="validation")
    batch_data["test"] = model_mgr.get_batch_data(
        data=datasets["test"], cols=config["COLUMNS"], shuffle=False,
        batch_size=config["HYPER_PARAMS"]["BATCH_SIZE"], is_test=True,
        workers=cpu_count(), label="test")

    # Prepare model
    model_mgr.prepare_model()

    # Train model
    model_mgr.train_model(
        train_data=batch_data["train"],
        score_after=config["SCORE_AFTER"], num_features=num_features,
        epochs=config["HYPER_PARAMS"]["EPOCHS"],
        learning_rate=config["HYPER_PARAMS"]["LEARNING_RATE"],
        batch_size=config["HYPER_PARAMS"]["BATCH_SIZE"],
        optimizer=config["HYPER_PARAMS"]["OPTIMIZER"],
        momentum=config["HYPER_PARAMS"]["MOMENTUM"],
        metric=config["HYPER_PARAMS"]["METRIC"])

    # Validate model
    model_mgr.validate_model(valid_data=batch_data["validation"],
                             metric=config["HYPER_PARAMS"]["METRIC"])

    # Test model
    results = model_mgr.test_model(
        test_loader=batch_data["test"], features=num_features,
        batch_size=config["HYPER_PARAMS"]["BATCH_SIZE"])

    # Save results
    model_mgr.save_results(
        cols={"id": config["COLUMNS"]["ID"],
              "target": config["COLUMNS"]["TARGET"]},
        data={"id": datasets["test"][config["columns"]["id"]].values,
              "target": results}, path=config["PATH"]["RESULT"])

    # Save model
    model_mgr.save_model(path={
        "params": config["PATH"]["MODEL"]["PARAMS"],
        "arch": config["PATH"]["MODEL"]["PARAMS"]
    })
