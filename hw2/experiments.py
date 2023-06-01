import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from cs236781.train_results import FitResult, EpochResult

from .cnn import CNN, ResNet
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
}


def mlp_experiment(
        depth: int,
        width: int,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        dl_test: DataLoader,
        n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======
    hidden_dims = [width] * (depth - 1)
    hidden_dims.append(2)
    nonlins = ['relu'] * (depth - 1)
    nonlins.append('none')

    hp_optim = dict(lr=5e-3, weight_decay=1e-2, momentum=0.85)

    mlp = MLP(in_dim=2, dims=hidden_dims, nonlins=nonlins)
    model = BinaryClassifier(mlp)

    trainer = ClassifierTrainer(model, torch.nn.CrossEntropyLoss(),
                                torch.optim.SGD(params=model.parameters(), **hp_optim))
    fit_res: FitResult = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0)

    optimal_thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
    model.threshold = optimal_thresh

    test_epoch_res: EpochResult = trainer.test_epoch(dl_test, verbose=False)
    valid_acc = fit_res.test_acc[-1]
    test_acc = test_epoch_res.accuracy
    # ========================
    return model, optimal_thresh, valid_acc, test_acc


def cnn_experiment(
        run_name,
        out_dir="./results",
        seed=None,
        device=None,
        # Training params
        bs_train=128,
        bs_test=None,
        batches=100,
        epochs=100,
        early_stopping=3,
        checkpoints=None,
        lr=1e-3,
        reg=1e-3,
        # Model params
        filters_per_layer=[64],
        layers_per_block=2,
        pool_every=2,
        hidden_dims=[1024],
        model_type="cnn",
        # You can add extra configuration for your experiments here
        cross_validation=False,
        **kw,
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=False)
    num_out_classes = 10  # CIFAR-10 is used for experiments and has 10 classes.
    in_size = ds_train[0][0].shape
    channels = _create_channels_cnn_experiment(layers_per_block, filters_per_layer)
    conv_params = dict(kernel_size=3, padding=1)
    pooling_params = dict(kernel_size=2, padding=1)
    loss_fn = torch.nn.CrossEntropyLoss()

    if cross_validation:
        val_size = 5000
        train_size = len(ds_train) - val_size
        cross_validation_train_set, cross_validation_val_set = random_split(ds_train, [train_size, val_size])

        # Define the values to be tested for pool_every and hidden_dims
        pool_every_values = [1, 2, 3]  # Example values, adjust as needed
        hidden_dims_values = [[25], [50], [100], [256], [512], [1024]]  # Example values, adjust as needed

        best_accuracy = 0.0
        best_pool_every = None
        best_hidden_dims = None

        for pool_every in pool_every_values:
            for hidden_dims in hidden_dims_values:
                # Train the model with the current pool_every and hidden_dims
                fit_res = None
                fit_res = create_model_and_fit(batches, channels, checkpoints, conv_params, device, dl_test, dl_train,
                                               early_stopping, epochs, fit_res, hidden_dims, in_size, loss_fn, lr,
                                               model_cls, num_out_classes, pool_every, pooling_params, reg)

                # Evaluate the model on the test set
                test_acc = fit_res.results.test_acc[-1]  # Assuming last accuracy is the final accuracy

                # Check if the current configuration is better than the previous best
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_pool_every = pool_every
                    best_hidden_dims = hidden_dims

        # Save the best configuration to a file
        best_config = {
            "pool_every": best_pool_every,
            "hidden_dims": best_hidden_dims,
        }
        save_best_config(run_name, out_dir, best_config)

    else:
        fit_res = create_model_and_fit(batches, channels, checkpoints, conv_params, device, dl_test, dl_train,
                                       early_stopping, epochs, fit_res, hidden_dims, in_size, loss_fn, lr,
                                       model_cls, num_out_classes, pool_every, pooling_params, reg)
        # ========================

        save_experiment(run_name, out_dir, cfg, fit_res)


def create_model_and_fit(batches, channels, checkpoints, conv_params, device, dl_test, dl_train, early_stopping, epochs,
                         fit_res, hidden_dims, in_size, loss_fn, lr, model_cls, num_out_classes, pool_every,
                         pooling_params, reg):
    model = model_cls(in_size, num_out_classes, channels, pool_every, hidden_dims, conv_params=conv_params,
                      pooling_params=pooling_params)
    print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=reg)
    classifier = ArgMaxClassifier(model).to(device)
    trainer = ClassifierTrainer(classifier, loss_fn, optimizer, device)
    fit_res = trainer.fit(dl_train, dl_test, epochs, checkpoints, early_stopping,
                          max_batches=batches)  # max_batches is used inside _foreach_batch in training.
    return fit_res


def _create_channels_cnn_experiment(layers_per_block, filters_per_layer):
    channels = []
    for k in filters_per_layer:
        channels.append([k] * layers_per_block)
    flattened_channels = itertools.chain.from_iterable(channels)
    return list(flattened_channels)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def save_best_config(run_name, out_dir, best_config):
    output_filename = f"{os.path.join(out_dir, run_name)}_best_config.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"*** Best configuration saved to: {output_filename}")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    # # Experiment 1_2
    # ks = [[32], [64], [128]]
    # ls = [8]
    # for l in ls:
    #     for k in ks:
    #         cnn_experiment(run_name='exp1_2', filters_per_layer=k, layers_per_block=l, pool_every=4,
    #                        hidden_dims=[512]*2, lr=1e-2, reg=0, early_stopping=5)

    # #Experiment 1_3
    # ks = [[64, 128]]
    # ls = [2, 3]
    # for l in ls:
    #     for k in ks:
    #         cnn_experiment(run_name='exp1_3', filters_per_layer=k, layers_per_block=l, pool_every=2,
    #                        hidden_dims=[100], lr=1e-3, early_stopping=100)

    # #Experimnt 1_4_1
    # ks = [[32]]
    # ls = [8, 16, 32]
    # for l in ls:
    #     for k in ks:
    #         cnn_experiment(run_name='exp1_3', filters_per_layer=k, layers_per_block=l, pool_every=4,
    #                        hidden_dims=[512]*2, lr=1e-3, early_stopping=5, model_type='resnet')

    #Experiment_1_4_2
    ks = [[64, 128, 256]]
    ls = [2, 4, 8]
    for l in ls:
        for k in ks:
            cnn_experiment(run_name='exp1_4', filters_per_layer=k, layers_per_block=l, pool_every=2,
                           hidden_dims=[512]*2, lr=1e-3, early_stopping=5, model_type='resnet')
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
