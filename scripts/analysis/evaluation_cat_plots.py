import os
import shutil
import argparse
import itertools

import numpy

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from sklearn.metrics import ConfusionMatrixDisplay

####

import goripy.file.json
import goripy.conf.metrics
import goripy.conf.aggs

####

from gorideep.utils.metadata import CategoryMetadata


        
#############################
# COMPUTING AUXILIARY METHODS
#############################



def metric_fun_factory(
    metric_fun_name,
    **kwargs
):

    if metric_fun_name == "prec":
        return goripy.conf.metrics.prec_metric_fun

    if metric_fun_name == "rec":
        return goripy.conf.metrics.rec_metric_fun
        
    if metric_fun_name == "acc":
        return goripy.conf.metrics.acc_metric_fun
        
    if metric_fun_name == "f1":
        return goripy.conf.metrics.f1_metric_fun
        
    if metric_fun_name == "f1b":
        return lambda tp, fp, fn, tn: \
            goripy.conf.metrics.f1b_metric_fun(
                tp, fp, fn, tn,
                b=kwargs["beta"]
            )

    raise ValueError("Unrecognized metric function name: \"{:s}\"".format(
        metric_fun_name
    ))



############################
# PLOTTING AUXILIARY METHODS
############################



def plot_cat_conf_mat(
    cat_conf_mat,
    cat_metadata,
    group_name,
    pn_normalized=True,
    plot_filename=None
):

    num_cats = cat_metadata.get_num_cats()

    #

    plt.figure(
        figsize=(
            1.50 + (num_cats * 0.50),
            1.50 + (num_cats * 0.50)
        )
    )

    #

    if pn_normalized:
        cat_conf_mat = \
            cat_conf_mat.astype("float") / numpy.sum(cat_conf_mat, axis=1)[:, None]

    cm_disp = ConfusionMatrixDisplay(
        confusion_matrix=cat_conf_mat
    )

    cm_dist_im_kw = {
        "vmin": 0.0,
        "vmax": 1.0
    }

    cm_disp.plot(
        ax=plt.gca(),
        colorbar=False,
        values_format="d" if not pn_normalized else ".3f",
        im_kw=cm_dist_im_kw
    )

    #

    cat_plot_name_list = [
        cat_name.replace(", ", "\n")
        for cat_name in cat_metadata.cat_name_list
    ]

    try:
        idx = cat_plot_name_list.index("headband\nhead covering\nhair accessory")
        cat_plot_name_list[idx] = "head\naccessory"
    except:
        pass

    plt.xticks(
        numpy.arange(num_cats),
        cat_plot_name_list,
        rotation=60
    )

    plt.yticks(
        numpy.arange(num_cats),
        cat_plot_name_list
    )

    suptitle = ""
    suptitle += "Confusion matrix"
    if pn_normalized:
        suptitle += " "
        suptitle += "(pn_normalized)"
    suptitle += "\n"
    suptitle += "Category: {:s}".format(
        cat_metadata._cat_subset_name,
    )
    suptitle += "\n"
    suptitle += "Group: {:s}".format(group_name)

    plt.suptitle(suptitle, y=0.95)

    #

    if plot_filename is None:
        plt.show()
    else:
        plt.savefig(plot_filename, bbox_inches="tight", dpi=160)
        plt.close()



def plot_cat_conf_aggs(
    cat_conf_aggs,
    cat_metadata,
    group_name,
    pn_normalized=True,
    plot_filename=None
):

    num_cats = cat_metadata.get_num_cats()

    #

    plt.figure(
        figsize=(
            num_cats * 0.75,
            4 * 0.75
        )
    )

    #

    pn_norm_cat_conf_aggs =\
        goripy.conf.aggs.pn_normalize_conf_aggs(cat_conf_aggs)

    plt.imshow(
        pn_norm_cat_conf_aggs.T,
        vmin=0.0,
        vmax=1.0
    )

    #

    if pn_normalized:
        text_color_thr = 0.5
    else:
        text_color_thr = numpy.max(cat_conf_aggs) * 0.5

    for x_idx, y_idx in itertools.product(range(num_cats), range(4)):
        
        if pn_normalized:
            value = pn_norm_cat_conf_aggs[x_idx, y_idx]
            value_str = "{:.3f}".format(value)
        else:
            value = cat_conf_aggs[x_idx, y_idx]
            value_str = "{:d}".format(value)

        text_color = "white" if value < text_color_thr else "black"

        plt.text(
            x_idx, y_idx, value_str,
            ha="center", va="center", weight="bold", color=text_color
        )

    #

    cat_plot_name_list = [
        cat_name.replace(", ", "\n")
        for cat_name in cat_metadata.cat_name_list
    ]

    try:
        idx = cat_plot_name_list.index("headband\nhead covering\nhair accessory")
        cat_plot_name_list[idx] = "head\naccessory"
    except:
        pass

    plt.xticks(
        numpy.arange(num_cats),
        cat_plot_name_list,
        rotation=60
    )

    plt.yticks(
        numpy.arange(4),
        ["TP", "FP", "FN", "TN"]
    )

    suptitle = ""
    suptitle += "Confusion aggregates"
    if pn_normalized:
        suptitle += " "
        suptitle += "(pn_normalized)"
    suptitle += "\n"
    suptitle += "Category: {:s}".format(
        cat_metadata._cat_subset_name,
    )
    suptitle += "\n"
    suptitle += "Group: {:s}".format(group_name)

    plt.suptitle(suptitle, y=1.20)

    #

    if plot_filename is None:
        plt.show()
    else:
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close()



def plot_cat_conf_metric(
    cat_conf_aggs,
    metric_fun,
    cat_metadata,
    group_name,
    metric_name,
    pn_normalized=True,
    show_numbers=False,
    plot_filename=None
):

    num_cats = cat_metadata.get_num_cats()

    #

    if pn_normalized:
        cat_conf_aggs = goripy.conf.aggs.pn_normalize_conf_aggs(cat_conf_aggs)
    
    metric_arr = goripy.conf.aggs.compute_conf_metric_arr(cat_conf_aggs, metric_fun)

    metric_micro_avg = goripy.conf.aggs.compute_conf_metric_avg(metric_arr, cat_conf_aggs, "micro")
    metric_macro_avg = goripy.conf.aggs.compute_conf_metric_avg(metric_arr, cat_conf_aggs, "macro")

    #

    plt.figure(
        figsize=(
            1.50 + (num_cats * 0.75),
            5
        )
    )

    plt.bar(numpy.arange(num_cats), metric_arr, zorder=3)

    #

    if show_numbers:

        plt.ylim(0, 1.06)
        text_y_diff = 0.03
        
        for cat_idx, metric in zip(numpy.arange(num_cats), metric_arr):

            if numpy.isnan(metric): continue

            plt.text(
                cat_idx, metric + text_y_diff, "{:.3f}".format(metric),
                ha="center", va="center", size=10
            )

    #

    invalid_cat_idx_arr = numpy.argwhere(numpy.isnan(metric_arr)).flatten()

    plt.scatter(
        invalid_cat_idx_arr, [0.05] * len(invalid_cat_idx_arr),
        s=150, c="red", marker="X"
    )

    #

    x0, x1 = plt.xlim()
    plt.xlim(
        x0 - (1 / num_cats),
        x1 + (1 / num_cats)
    )

    #

    cat_plot_name_list = [
        cat_name.replace(", ", "\n")
        for cat_name in cat_metadata.cat_name_list
    ]

    try:
        idx = cat_plot_name_list.index("headband\nhead covering\nhair accessory")
        cat_plot_name_list[idx] = "head\naccessory"
    except:
        pass

    plt.xticks(
        numpy.arange(num_cats),
        labels=cat_plot_name_list,
        rotation=60
    )

    plt.grid(axis="y")

    suptitle = ""
    suptitle += "Metric: {:s}".format(metric_name)
    if pn_normalized:
        suptitle += " (pn_normalized)"
    suptitle += "\n"
    suptitle += "Category: {:s}".format(
        cat_metadata._cat_subset_name,
    )
    suptitle += "\n"
    suptitle += "Group: {:s}".format(group_name)
    suptitle += "\n"
    suptitle += "Micro avg: {:.3f} | Macro avg: {:.3f}".format(
        metric_micro_avg,
        metric_macro_avg
    )

    plt.suptitle(suptitle, y=1.10)

    #

    if plot_filename is None:
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
        "eval_name",
        help="name of the evaluation run"
    )
    
    command_args = parser.parse_args()

    ana_name = command_args.ana_name
    eval_name = command_args.eval_name


    #
    # Create analysis result directories
    #


    ana_settings_dirname = os.path.join(
        os.environ["GORIDEEPTRAIN_DATA_HOME"], "analysis_settings", ana_name
    )
    
    ana_results_dirname = os.path.join(
        os.environ["GORIDEEPTRAIN_DATA_HOME"], "analysis_results", ana_name
    )

    if os.path.exists(ana_results_dirname): shutil.rmtree(ana_results_dirname)
    os.makedirs(ana_results_dirname, exist_ok=True)

    ana_settings_settings_dirname = os.path.join(ana_settings_dirname, "settings")
    ana_results_settings_dirname = os.path.join(ana_results_dirname, "settings")
    shutil.copytree(ana_settings_settings_dirname, ana_results_settings_dirname)

    ana_data_dirname = os.path.join(ana_results_dirname, "data")
    os.mkdir(ana_data_dirname)

    cat_conf_mat_plots_dirname = os.path.join(ana_data_dirname, "cat_conf_mat_plots")
    cat_conf_aggs_plots_dirname = os.path.join(ana_data_dirname, "cat_conf_aggs_plots")
    cat_conf_metric_plots_dirname = os.path.join(ana_data_dirname, "cat_conf_metric_plots")

    os.mkdir(cat_conf_mat_plots_dirname)
    os.mkdir(cat_conf_aggs_plots_dirname)
    os.mkdir(cat_conf_metric_plots_dirname)


    #
    # Load analysis settings
    #

    
    ana_metrics_settings = goripy.file.json.load_json(os.path.join(
        ana_results_settings_dirname, "metrics.json"
    ))

    ana_groups_settings = goripy.file.json.load_json(os.path.join(
        ana_results_settings_dirname, "groups.json"
    ))

    ana_plots_settings = goripy.file.json.load_json(os.path.join(
        ana_results_settings_dirname, "plots.json"
    ))


    #
    # Load evaluation data
    #


    eval_results_dirname = os.path.join(
        os.environ["GORIDEEPTRAIN_DATA_HOME"], "evaluation_results", eval_name
    )

    # Load evaluation data

    eval_data_filename = os.path.join(
        eval_results_dirname, "eval_data.json"
    )

    eval_data = goripy.file.json.load_json(eval_data_filename)

    # Load metadata settings

    metadata_settings_filename = os.path.join(
        eval_results_dirname, "settings", "metadata.json"
    )

    metadata_settings = goripy.file.json.load_json(metadata_settings_filename)

    cat_metadata = CategoryMetadata(metadata_settings["cat_subset_name"])

    # Load dataset settings

    dataset_name_list = list(eval_data["status"]["progress"].keys())
    split_str_list = ["train", "val", "test"]


    #
    # Pre-compute aggregates for each dataset and split
    #


    dataset_split_data_dict = {
        dataset_name: {
            split_str: None
            for split_str in split_str_list
        }
        for dataset_name in dataset_name_list
    }

    pred_cat_logits_dirname = os.path.join(
        eval_results_dirname, "data", "pred_cat_logits"
    )

    target_cat_probs_dirname = os.path.join(
        eval_results_dirname, "data", "target_cat_probs"
    )

    cat_weights_dirname = os.path.join(
        eval_results_dirname, "data", "cat_weights"
    )

    for dataset_name, split_str in itertools.product(
        dataset_name_list,
        split_str_list
    ):

        # Check existence and load results

        pred_cat_logits_filename = os.path.join(
            pred_cat_logits_dirname, dataset_name, "{:s}.pt".format(split_str)
        )
        
        target_cat_probs_filename = os.path.join(
            target_cat_probs_dirname, dataset_name, "{:s}.pt".format(split_str)
        )
        
        cat_weights_filename = os.path.join(
            cat_weights_dirname, dataset_name, "{:s}.pt".format(split_str)
        )

        if not os.path.exists(pred_cat_logits_filename): continue
        if not os.path.exists(target_cat_probs_filename): continue
        if not os.path.exists(cat_weights_filename): continue

        dataset_split_data_dict[dataset_name][split_str] = {}

        pred_cat_logits_arrr = \
            torch.load(pred_cat_logits_filename, map_location="cpu").numpy()

        target_cat_probs_arrr = \
            torch.load(target_cat_probs_filename, map_location="cpu").numpy()
        
        cat_weights_arrr = \
            torch.load(cat_weights_filename, map_location="cpu").numpy()

        # Compute confusion matrix

        cat_conf_mat = numpy.zeros(
            shape=(cat_metadata.get_num_cats(), cat_metadata.get_num_cats()),
            dtype=numpy.uint32
        )

        pred_cat_idx_arr = numpy.argmax(pred_cat_logits_arrr * cat_weights_arrr, axis=1)
        target_cat_idx_arr = numpy.argmax(target_cat_probs_arrr, axis=1)
        
        for target_cat_idx, pred_cat_idx in zip(target_cat_idx_arr, pred_cat_idx_arr):
            cat_conf_mat[target_cat_idx, pred_cat_idx] += 1

        dataset_split_data_dict[dataset_name][split_str]["cat_conf_mat"] = cat_conf_mat
        
        # Compute confusion matrix aggregates (for each super-attribute)
        # Order: tp, fp, fn, tn

        cat_conf_aggs = numpy.empty(
            shape=(cat_metadata.get_num_cats(), 4),
            dtype=numpy.uint32
        )

        cat_arange = numpy.arange(cat_metadata.get_num_cats())

        cat_conf_mat_diag = cat_conf_mat[cat_arange, cat_arange]
        cat_conf_mat_col_sum = numpy.sum(cat_conf_mat, axis=0)
        cat_conf_mat_row_sum = numpy.sum(cat_conf_mat, axis=1)
        cat_conf_mat_sum = numpy.sum(cat_conf_mat)

        cat_conf_aggs[:, 0] = cat_conf_mat_diag
        cat_conf_aggs[:, 1] = cat_conf_mat_col_sum - cat_conf_mat_diag
        cat_conf_aggs[:, 2] = cat_conf_mat_row_sum - cat_conf_mat_diag
        cat_conf_aggs[:, 3] = \
            cat_conf_mat_sum + cat_conf_mat_diag \
            - cat_conf_mat_col_sum - cat_conf_mat_row_sum

        dataset_split_data_dict[dataset_name][split_str]["cat_conf_aggs"] = cat_conf_aggs


    #
    # Pre-compute plot data
    #


    # Prepare metric functions

    metric_fun_dict = {}

    for metric_name, metric_fun_info in ana_metrics_settings.items():

        metric_fun_dict[metric_name] = metric_fun_factory(
            metric_fun_info["metric_fun_name"],
            **(metric_fun_info.get("kwargs", {}))
        )

    # Compute evaluation summaries for each group

    group_data_dict = {}

    for group_name, group in ana_groups_settings.items():

        group_data_dict[group_name] = {}

        ## Fill confusion matrix and aggregates

        group_cat_conf_mat = numpy.zeros(
            shape=(cat_metadata.get_num_cats(), cat_metadata.get_num_cats()),
            dtype=numpy.uint32
        )

        group_cat_conf_aggs = numpy.zeros(
            shape=(cat_metadata.get_num_cats(), 4),
            dtype=numpy.uint32
        )

        for group_el in group:

            dataset_name = group_el["dataset"]
            split_str = group_el["split"]

            group_cat_conf_mat += \
                dataset_split_data_dict[dataset_name][split_str]["cat_conf_mat"]

            group_cat_conf_aggs += \
                dataset_split_data_dict[dataset_name][split_str]["cat_conf_aggs"]
            
        
        group_data_dict[group_name]["cat_conf_mat"] = group_cat_conf_mat
        group_data_dict[group_name]["cat_conf_aggs"] = group_cat_conf_aggs


    #
    # Create plots
    #


    # Category confusion matrix plots

    for plot_config in ana_plots_settings["cat_conf_mat"]:

        for group_name in plot_config["group_names"]:

            plot_name = group_name
            if plot_config["pn_normalized"]: plot_name += "___pn_norm"
            plot_filename = os.path.join(cat_conf_mat_plots_dirname, plot_name + ".jpg")

            plot_cat_conf_mat(
                group_data_dict[group_name]["cat_conf_mat"],
                cat_metadata,
                group_name,
                pn_normalized=plot_config["pn_normalized"],
                plot_filename=plot_filename
            )

    # Category confusion aggregate plots

    for plot_config in ana_plots_settings["cat_conf_aggs"]:

        for group_name in plot_config["group_names"]:

            plot_name = group_name
            if plot_config["pn_normalized"]: plot_name += "___pn_norm"
            plot_filename = os.path.join(cat_conf_aggs_plots_dirname, plot_name + ".jpg")

            plot_cat_conf_aggs(
                group_data_dict[group_name]["cat_conf_aggs"],
                cat_metadata,
                group_name,
                pn_normalized=plot_config["pn_normalized"],
                plot_filename=plot_filename
            )

    # Category confusion metric plots

    for plot_config in ana_plots_settings["cat_conf_metric"]:

        for group_name, metric_name in itertools.product(
            plot_config["group_names"],
            plot_config["metric_names"]
        ):

            plot_name = group_name + "___" + metric_name
            if plot_config["pn_normalized"]: plot_name += "___pn_norm"
            if plot_config["show_numbers"]: plot_name += "___nums"
            plot_filename = os.path.join(cat_conf_metric_plots_dirname, plot_name + ".jpg")

            plot_cat_conf_metric(
                group_data_dict[group_name]["cat_conf_aggs"],
                metric_fun_dict[metric_name],
                cat_metadata,
                group_name,
                metric_name,
                pn_normalized=plot_config["pn_normalized"],
                show_numbers=plot_config["show_numbers"],
                plot_filename=plot_filename
            )
