import os
import shutil
import argparse
import itertools

import numpy

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

import goripy.file.json


        
###################
# AUXILIARY METHODS
###################



def compute_epoch_total_loss_arr(
    loss_reg_key_list,
    loss_reg_epoch_data_pool,
    loss_weight_arr_dict,
    last_epoch_num,
    split_str,
    weighted=True,
    per_item=False
):

    arr_len = last_epoch_num
    if split_str == "train": arr_len -= 1

    all_epoch_total_loss_arr = numpy.zeros(shape=(arr_len), dtype=float)
    all_epoch_total_items_arr = numpy.zeros(shape=(arr_len), dtype=float)

    for loss_reg_key in loss_reg_key_list:

        loss_reg_epoch_data = loss_reg_epoch_data_pool[split_str][loss_reg_key]

        epoch_total_loss_arr = loss_reg_epoch_data["epoch_total_loss_arr"].copy()
        epoch_total_items_arr = loss_reg_epoch_data["epoch_total_items_arr"].copy()

        if not weighted:

            loss_wt_arr = loss_weight_arr_dict[loss_reg_key]
            if split_str == "train": loss_wt_arr = loss_wt_arr[1:]
            epoch_total_loss_arr /= loss_wt_arr

        all_epoch_total_loss_arr += epoch_total_loss_arr
        all_epoch_total_items_arr += epoch_total_items_arr
    
    if per_item:

        all_epoch_total_loss_arr /= all_epoch_total_items_arr

    return all_epoch_total_loss_arr



def compute_epoch_spec_loss_arr(
    loss_reg_key,
    loss_reg_epoch_data_pool,
    loss_weight_arr_dict,
    last_epoch_num,
    split_str,
    weighted=True,
    per_item=True
):

    loss_reg_epoch_data = loss_reg_epoch_data_pool[split_str][loss_reg_key]

    loss_arr = loss_reg_epoch_data["epoch_total_loss_arr"].copy()
    num_item_arr = loss_reg_epoch_data["epoch_total_items_arr"].copy()

    if not weighted:

        loss_wt_arr = loss_weight_arr_dict[loss_reg_key]
        if split_str == "train": loss_wt_arr = loss_wt_arr[1:]
        loss_arr /= loss_weight_arr_dict[loss_reg_key]

    valid_flag_arr = num_item_arr != 0

    if per_item:
        loss_arr[valid_flag_arr] /= num_item_arr[valid_flag_arr]
    
    return loss_arr, valid_flag_arr



def compute_epoch_group_loss_arr(
    loss_reg_key_list,
    loss_reg_epoch_data_pool,
    loss_weight_arr_dict,
    last_epoch_num,
    split_str,
    weighted=True,
    per_item=True
):

    arr_size = last_epoch_num - 1 if split_str == "train" else last_epoch_num

    loss_arr = numpy.zeros(shape=(arr_size), dtype=float)
    num_item_arr = numpy.zeros(shape=(arr_size), dtype=numpy.uint32)

    for loss_reg_key in loss_reg_key_list:

        loss_reg_epoch_data = loss_reg_epoch_data_pool[split_str][loss_reg_key]

        spec_loss_arr = loss_reg_epoch_data["epoch_total_loss_arr"].copy()
        spec_num_item_arr = loss_reg_epoch_data["epoch_total_items_arr"].copy().astype(numpy.uint32)

        if not weighted:

            loss_wt_arr = loss_weight_arr_dict[loss_reg_key]
            if split_str == "train": loss_wt_arr = loss_wt_arr[1:]
            spec_loss_arr /= loss_weight_arr_dict[loss_reg_key]
        
        loss_arr += spec_loss_arr
        num_item_arr += spec_num_item_arr

    valid_flag_arr = num_item_arr != 0
    
    if per_item:
        loss_arr[valid_flag_arr] /= num_item_arr[valid_flag_arr]
    
    return loss_arr, valid_flag_arr



def generate_epoch_total_loss_plot(
    train_loss_arr,
    val_loss_arr,
    last_epoch_num,
    plot_filename=None
):

    plt.figure(figsize=(10, 5))

    plt.plot(
        numpy.arange(1, last_epoch_num),
        train_loss_arr,
        marker=".",
        label="Train"
    )
    
    plt.plot(
        numpy.arange(0, last_epoch_num),
        val_loss_arr,
        marker=".",
        label="Val"
    )

    plt.grid()
    plt.xlim(-1, last_epoch_num + 1)
    plt.yscale("log")
    plt.legend()

    suptitle = ""
    suptitle += "Total Loss"
    suptitle += "\n"
    suptitle += "Min Train Loss: {:.2e}".format(numpy.min(train_loss_arr))
    suptitle += " | "
    suptitle += "Min Val Loss: {:.2e}".format(numpy.min(val_loss_arr))

    plt.suptitle(suptitle)

    #

    if plot_filename is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close()



def generate_epoch_spec_group_loss_plot(
    loss_name,
    train_loss_arr,
    train_valid_flag_arr,
    val_loss_arr,
    val_valid_flag_arr,
    last_epoch_num,
    plot_filename=None
):
    
    all_train_invalid_loss = numpy.all(~train_valid_flag_arr)
    all_val_invalid_loss = numpy.all(~val_valid_flag_arr)

    plt.figure(figsize=(10, 5))

    if not all_train_invalid_loss:
        plt.plot(
            numpy.arange(1, last_epoch_num)[train_valid_flag_arr],
            train_loss_arr[train_valid_flag_arr],
            marker=".",
            label="Train"
        )
    
    if not all_val_invalid_loss:
        plt.plot(
            numpy.arange(0, last_epoch_num)[val_valid_flag_arr],
            val_loss_arr[val_valid_flag_arr],
            marker=".",
            label="Val"
        )

    # for epoch_num in numpy.arange(1, last_epoch_num)[~train_valid_flag_arr]:
    #     plt.axvline(
    #         epoch_num-0.05,
    #         linestyle="solid",
    #         linewidth=5,
    #         color="tab:blue",
    #         alpha=0.25
    #     )

    # for epoch_num in numpy.arange(0, last_epoch_num)[~val_valid_flag_arr]:
    #     plt.axvline(
    #         epoch_num+0.05,
    #         linestyle="solid",
    #         linewidth=5,
    #         color="tab:orange",
    #         alpha=0.25
    #     )

    plt.grid()
    plt.xlim(-1, last_epoch_num + 1)
    plt.yscale("log")

    if not (all_train_invalid_loss and all_val_invalid_loss):
        plt.legend()

    suptitle = ""
    suptitle += "Loss: {:s}".format(loss_name)
    suptitle += "\n"
    suptitle += "Min Train Loss: {:.2e}".format(numpy.nanmin(train_loss_arr))
    suptitle += " | "
    suptitle += "Min Val Loss: {:.2e}".format(numpy.nanmin(val_loss_arr))

    plt.suptitle(suptitle)

    #

    if plot_filename is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close()



############
# ENTRYPOINT
############



if __name__ == "__main__":
    

    #
    # Parse command args
    #


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "ana_name",
        help="name of the experiment run"
    )
    
    parser.add_argument(
        "exp_name",
        help="name of the experiment run"
    )
    
    command_args = parser.parse_args()


    #
    # Create analysis result directories
    #


    ana_settings_dirname = os.path.join(
        os.environ["GORIDEEPTRAIN_DATA_HOME"], "analysis_settings", command_args.ana_name
    )
    
    ana_results_dirname = os.path.join(
        os.environ["GORIDEEPTRAIN_DATA_HOME"], "analysis_results", command_args.ana_name
    )

    if os.path.exists(ana_results_dirname): shutil.rmtree(ana_results_dirname)
    os.makedirs(ana_results_dirname, exist_ok=True)

    ana_settings_settings_dirname = os.path.join(ana_settings_dirname, "settings")
    ana_results_settings_dirname = os.path.join(ana_results_dirname, "settings")
    shutil.copytree(ana_settings_settings_dirname, ana_results_settings_dirname)

    os.mkdir(os.path.join(ana_results_dirname, "data"))
    os.mkdir(os.path.join(ana_results_dirname, "data", "spec_loss_plots"))
    os.mkdir(os.path.join(ana_results_dirname, "data", "group_loss_plots"))


    #
    # Load analysis settings
    #


    ana_groups_settings = goripy.file.json.load_json(os.path.join(
        ana_results_settings_dirname, "groups.json"
    ))

    ana_plots_settings = goripy.file.json.load_json(os.path.join(
        ana_results_settings_dirname, "plots.json"
    ))


    #
    # Load experiment data
    #


    exp_results_dirname = os.path.join(
        os.environ["GORIDEEPTRAIN_DATA_HOME"], "experiment_results", command_args.exp_name
    )

    # Load experiment data

    exp_data_filename = os.path.join(exp_results_dirname, "exp_data.json")
    exp_data = goripy.file.json.load_json(exp_data_filename)

    # Load loss register settings

    exp_settings_dirname = os.path.join(exp_results_dirname, "settings")
    exp_settings_filename = os.path.join(exp_settings_dirname, "loss_registers.json")
    loss_reg_settings = goripy.file.json.load_json(exp_settings_filename)

    loss_reg_key_list = loss_reg_settings["loss_reg_key_list"]

    # Load loss register data

    loss_reg_epoch_data_pool = {"train": {}, "val": {}}
    last_epoch_num = -1 

    loss_reg_data_dirname = os.path.join(exp_results_dirname, "data", "loss_reg_data")

    for split_str, loss_reg_key in itertools.product(
        ["train", "val"],
        loss_reg_key_list
    ):

        loss_reg_epoch_data_filename = os.path.join(loss_reg_data_dirname, split_str, loss_reg_key, "epoch_data.npz")
        loss_reg_epoch_data_npz = numpy.load(loss_reg_epoch_data_filename)
        loss_reg_epoch_data_pool[split_str][loss_reg_key] = dict(loss_reg_epoch_data_npz)

        num_loss_epochs = loss_reg_epoch_data_npz["epoch_total_loss_arr"].shape[0]
        if split_str == "train": num_loss_epochs += 1
        last_epoch_num = max(last_epoch_num, num_loss_epochs)

        del loss_reg_epoch_data_npz

    # Load loss weight data

    loss_weight_data_dirname = os.path.join(exp_results_dirname, "data", "loss_weight_data")

    loss_weight_arr_dict = {
        loss_reg_key: numpy.empty(shape=(last_epoch_num), dtype=float)
        for loss_reg_key in loss_reg_key_list
    }

    for epoch_num in range(last_epoch_num):

        loss_weights_dict = goripy.file.json.load_json(os.path.join(
            loss_weight_data_dirname, "loss_weights_{:06d}.json".format(epoch_num)
        ))
        
        for loss_reg_key in loss_reg_key_list:
            loss_weight_arr_dict[loss_reg_key][epoch_num] = loss_weights_dict[loss_reg_key]

    
    #
    # Generate loss plots
    #


    # Total loss plot

    train_loss_arr = compute_epoch_total_loss_arr(
        loss_reg_key_list,
        loss_reg_epoch_data_pool,
        loss_weight_arr_dict,
        last_epoch_num=last_epoch_num,
        split_str="train",
        weighted=ana_plots_settings["total_loss"]["weighted"],
        per_item=ana_plots_settings["total_loss"]["per_item"]
    )

    val_loss_arr = compute_epoch_total_loss_arr(
        loss_reg_key_list,
        loss_reg_epoch_data_pool,
        loss_weight_arr_dict,
        last_epoch_num=last_epoch_num,
        split_str="val",
        weighted=ana_plots_settings["total_loss"]["weighted"],
        per_item=ana_plots_settings["total_loss"]["per_item"]
    )

    plot_filename = os.path.join(ana_results_dirname, "data", "total_loss_plot.jpg")

    generate_epoch_total_loss_plot(
        train_loss_arr,
        val_loss_arr,
        last_epoch_num,
        plot_filename
    )

    # Specific loss plots

    for loss_reg_key in loss_reg_key_list:

        train_loss_arr, train_valid_flag_arr = compute_epoch_spec_loss_arr(
            loss_reg_key,
            loss_reg_epoch_data_pool,
            loss_weight_arr_dict,
            last_epoch_num=last_epoch_num,
            split_str="train",
            weighted=ana_plots_settings["spec_loss"]["weighted"],
            per_item=ana_plots_settings["spec_loss"]["per_item"]
        )

        val_loss_arr, val_valid_flag_arr = compute_epoch_spec_loss_arr(
            loss_reg_key,
            loss_reg_epoch_data_pool,
            loss_weight_arr_dict,
            last_epoch_num=last_epoch_num,
            split_str="val",
            weighted=ana_plots_settings["spec_loss"]["weighted"],
            per_item=ana_plots_settings["spec_loss"]["per_item"]
        )

        plot_filename = os.path.join(ana_results_dirname, "data", "spec_loss_plots", "{:s}__plot.jpg".format(
            loss_reg_key
        ))

        generate_epoch_spec_group_loss_plot(
            loss_reg_key,
            train_loss_arr,
            train_valid_flag_arr,
            val_loss_arr,
            val_valid_flag_arr,
            last_epoch_num,
            plot_filename
        )

    # Group loss plots

    for group_loss_item in ana_plots_settings["group_loss"]:

        group_name = group_loss_item["group_name"]

        train_loss_arr, train_valid_flag_arr = compute_epoch_group_loss_arr(
            ana_groups_settings[group_name],
            loss_reg_epoch_data_pool,
            loss_weight_arr_dict,
            last_epoch_num=last_epoch_num,
            split_str="train",
            weighted=group_loss_item["weighted"],
            per_item=group_loss_item["per_item"]
        )

        val_loss_arr, val_valid_flag_arr = compute_epoch_group_loss_arr(
            ana_groups_settings[group_name],
            loss_reg_epoch_data_pool,
            loss_weight_arr_dict,
            last_epoch_num=last_epoch_num,
            split_str="val",
            weighted=group_loss_item["weighted"],
            per_item=group_loss_item["per_item"]
        )

        plot_filename = os.path.join(ana_results_dirname, "data", "group_loss_plots", "{:s}__plot.jpg".format(
            group_name
        ))

        generate_epoch_spec_group_loss_plot(
            loss_reg_key,
            train_loss_arr,
            train_valid_flag_arr,
            val_loss_arr,
            val_valid_flag_arr,
            last_epoch_num,
            plot_filename
        )
