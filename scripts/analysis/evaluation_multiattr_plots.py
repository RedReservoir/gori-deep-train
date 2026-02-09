import os
import shutil
import argparse
import itertools

import numpy

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

####

import goripy.file.json
import goripy.conf.metrics
import goripy.conf.aggs

####

from gorideep.utils.metadata import MultiAttributeMetadata


        
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



def plot_supattr_conf_aggs(
    supattr_conf_aggs,
    multiattr_metadata,
    supattr_name,
    group_name,
    pn_normalized=True,
    plot_filename=None
):

    supattr_idx = multiattr_metadata.supattr_name_to_idx_dict[supattr_name]
    supattr_size = multiattr_metadata.supattr_size_list[supattr_idx]

    #

    plt.figure(
        figsize=(
            supattr_size * 0.75,
            4 * 0.75
        )
    )
    #

    pn_norm_supattr_conf_aggs =\
        goripy.conf.aggs.pn_normalize_conf_aggs(supattr_conf_aggs)

    plt.imshow(
        pn_norm_supattr_conf_aggs.T,
        vmin=0.0,
        vmax=1.0
    )

    #

    if pn_normalized:
        text_color_thr = 0.5
    else:
        text_color_thr = numpy.max(supattr_conf_aggs) * 0.5

    for x_idx, y_idx in itertools.product(range(supattr_size), range(4)):
        
        if pn_normalized:
            value = pn_norm_supattr_conf_aggs[x_idx, y_idx]
            value_str = "{:.3f}".format(value)
        else:
            value = supattr_conf_aggs[x_idx, y_idx]
            value_str = "{:d}".format(value)

        text_color = "white" if value < text_color_thr else "black"

        plt.text(
            x_idx, y_idx, value_str,
            ha="center", va="center", weight="bold", color=text_color
        )

    #

    supattr_attr_plot_name_list = []
    for supattr_attr_idx in range(supattr_size):
        attr_idx = multiattr_metadata.supattr_attr_idxs_to_attr_idx_list_dict[supattr_idx][supattr_attr_idx]
        attr_name = multiattr_metadata.attr_name_list[attr_idx]
        supattr_attr_plot_name_list.append(attr_name)

    plt.xticks(
        numpy.arange(supattr_size),
        supattr_attr_plot_name_list,
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
    suptitle += "Multi-attribute: {:s} - {:s}".format(
        multiattr_metadata._multiattr_subset_name,
        supattr_name
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



def plot_supattr_conf_metric(
    supattr_conf_aggs,
    metric_fun,
    multiattr_metadata,
    supattr_name,
    group_name,
    metric_name,
    pn_normalized=True,
    show_numbers=False,
    plot_filename=None
):

    supattr_idx = multiattr_metadata.supattr_name_to_idx_dict[supattr_name]
    supattr_size = multiattr_metadata.supattr_size_list[supattr_idx]

    #

    if pn_normalized:
        supattr_conf_aggs = goripy.conf.aggs.pn_normalize_conf_aggs(supattr_conf_aggs)
    
    metric_arr = goripy.conf.aggs.compute_conf_metric_arr(supattr_conf_aggs, metric_fun)

    metric_micro_avg = goripy.conf.aggs.compute_conf_metric_avg(metric_arr, supattr_conf_aggs, "micro")
    metric_macro_avg = goripy.conf.aggs.compute_conf_metric_avg(metric_arr, supattr_conf_aggs, "macro")

    #

    plt.figure(
        figsize=(
            1.50 + (supattr_size * 0.75),
            5
        )
    )

    plt.bar(numpy.arange(supattr_size), metric_arr, zorder=3)

    #

    if show_numbers:

        plt.ylim(0, 1.06)
        text_y_diff = 0.03
        
        for supattr_attr_idx, metric in zip(numpy.arange(supattr_size), metric_arr):

            if numpy.isnan(metric): continue

            plt.text(
                supattr_attr_idx, metric + text_y_diff, "{:.3f}".format(metric),
                ha="center", va="center", size=10
            )

    #

    invalid_supattr_attr_idx_arr = numpy.argwhere(numpy.isnan(metric_arr)).flatten()

    plt.scatter(
        invalid_supattr_attr_idx_arr, [0.05] * len(invalid_supattr_attr_idx_arr),
        s=150, c="red", marker="X"
    )

    #

    x0, x1 = plt.xlim()
    plt.xlim(
        x0 - (1 / supattr_size),
        x1 + (1 / supattr_size)
    )

    #

    supattr_attr_plot_name_list = []
    for supattr_attr_idx in range(supattr_size):
        attr_idx = multiattr_metadata.supattr_attr_idxs_to_attr_idx_list_dict[supattr_idx][supattr_attr_idx]
        attr_name = multiattr_metadata.attr_name_list[attr_idx]
        supattr_attr_plot_name_list.append(attr_name)

    plt.xticks(
        numpy.arange(supattr_size),
        labels=supattr_attr_plot_name_list,
        rotation=60
    )

    plt.grid(axis="y")

    suptitle = ""
    suptitle += "Metric: {:s}".format(metric_name)
    if pn_normalized:
        suptitle += " (pn_normalized)"
    suptitle += "\n"
    suptitle += "Multi-attribute: {:s} - {:s}".format(
        multiattr_metadata._multiattr_subset_name,
        supattr_name
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

    multiattr_conf_aggs_plots_dirname = os.path.join(ana_data_dirname, "multiattr_conf_aggs_plots")
    multiattr_conf_metric_plots_dirname = os.path.join(ana_data_dirname, "multiattr_conf_metric_plots")

    os.mkdir(multiattr_conf_aggs_plots_dirname)
    os.mkdir(multiattr_conf_metric_plots_dirname)


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

    multiattr_metadata = MultiAttributeMetadata(metadata_settings["multiattr_subset_name"])

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

    pred_multiattr_logits_dirname = os.path.join(
        eval_results_dirname, "data", "pred_multiattr_logits"
    )

    target_multiattr_probs_dirname = os.path.join(
        eval_results_dirname, "data", "target_multiattr_probs"
    )

    multiattr_weights_dirname = os.path.join(
        eval_results_dirname, "data", "multiattr_weights"
    )

    for dataset_name, split_str in itertools.product(
        dataset_name_list,
        split_str_list
    ):

        # Check existence and load results

        pred_multiattr_logits_filename = os.path.join(
            pred_multiattr_logits_dirname, dataset_name, "{:s}.pt".format(split_str)
        )
        
        target_multiattr_probs_filename = os.path.join(
            target_multiattr_probs_dirname, dataset_name, "{:s}.pt".format(split_str)
        )
        
        multiattr_weights_filename = os.path.join(
            multiattr_weights_dirname, dataset_name, "{:s}.pt".format(split_str)
        )

        if not os.path.exists(pred_multiattr_logits_filename): continue
        if not os.path.exists(target_multiattr_probs_filename): continue
        if not os.path.exists(multiattr_weights_filename): continue

        dataset_split_data_dict[dataset_name][split_str] = {}

        pred_multiattr_logits_arrr_dict = {
            supattr_name: pred_multiattr_logits_tenn.numpy()
            for supattr_name, pred_multiattr_logits_tenn in \
            torch.load(pred_multiattr_logits_filename, map_location="cpu").items()
        }

        target_multiattr_probs_arrr_dict = {
            supattr_name: target_multiattr_probs_tenn.numpy()
            for supattr_name, target_multiattr_probs_tenn in \
            torch.load(target_multiattr_probs_filename, map_location="cpu").items()
        }
        
        multiattr_weights_arrr_dict = {
            supattr_name: multiattr_weights_tenn.numpy()
            for supattr_name, multiattr_weights_tenn in \
            torch.load(multiattr_weights_filename, map_location="cpu").items()
        }

        # Compute confusion matrix aggregates (for each super-attribute)
        # Order: tp, fp, fn, tn

        dataset_split_data_dict[dataset_name][split_str]["supattr_conf_aggs"] = {}

        for supattr_name, supattr_size in zip(
            multiattr_metadata.supattr_name_list,
            multiattr_metadata.supattr_size_list
        ):

            supattr_conf_aggs = numpy.zeros(
                shape=(supattr_size, 4),
                dtype=numpy.uint32
            )

            supattr_pred_attr_logits_arrr = pred_multiattr_logits_arrr_dict[supattr_name]
            supattr_target_attr_probs_arrr = target_multiattr_probs_arrr_dict[supattr_name]
            supattr_attr_weights_arrr = multiattr_weights_arrr_dict[supattr_name]

            supattr_pred_attr_flags_arrr = (supattr_pred_attr_logits_arrr >= 0.5)
            supattr_target_attr_flags_arrr = (supattr_target_attr_probs_arrr >= 0.5)
            supattr_attr_weight_flags_arrr = (supattr_attr_weights_arrr >= 0.5)

            supattr_conf_aggs[:, 0] =\
                numpy.sum(
                    supattr_pred_attr_flags_arrr & supattr_target_attr_flags_arrr & supattr_attr_weight_flags_arrr,
                    axis=0
                )
            
            supattr_conf_aggs[:, 1] =\
                numpy.sum(
                    supattr_pred_attr_flags_arrr & ~supattr_target_attr_flags_arrr & supattr_attr_weight_flags_arrr,
                    axis=0
                )
            
            supattr_conf_aggs[:, 2] =\
                numpy.sum(
                    (~supattr_pred_attr_flags_arrr) & (supattr_target_attr_flags_arrr) & supattr_attr_weight_flags_arrr,
                    axis=0
                )
            
            supattr_conf_aggs[:, 3] =\
                numpy.sum(
                    (~supattr_pred_attr_flags_arrr) & (~supattr_target_attr_flags_arrr) & supattr_attr_weight_flags_arrr,
                    axis=0
                )

            dataset_split_data_dict[dataset_name][split_str]["supattr_conf_aggs"][supattr_name] = supattr_conf_aggs


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

        ## Fill confusion aggregates

        group_data_dict[group_name]["supattr_conf_aggs"] = {}

        for supattr_name, supattr_size in zip(
            multiattr_metadata.supattr_name_list,
            multiattr_metadata.supattr_size_list
        ):

            group_supattr_conf_aggs = numpy.zeros(
                shape=(supattr_size, 4),
                dtype=numpy.uint32
            )

            for group_el in group:

                dataset_name = group_el["dataset"]
                split_str = group_el["split"]

                group_supattr_conf_aggs += \
                    dataset_split_data_dict[dataset_name][split_str]["supattr_conf_aggs"][supattr_name]
            
            group_data_dict[group_name]["supattr_conf_aggs"][supattr_name] = group_supattr_conf_aggs


    #
    # Create plots
    #


    # Category confusion aggregate plots

    for plot_config in ana_plots_settings["multiattr_conf_aggs"]:

        for group_name in plot_config["group_names"]:

            for supattr_name in multiattr_metadata.supattr_name_list:

                plot_name = group_name + "___" + supattr_name
                if plot_config["pn_normalized"]: plot_name += "___pn_norm"
                plot_filename = os.path.join(multiattr_conf_aggs_plots_dirname, plot_name + ".jpg")

                plot_supattr_conf_aggs(
                    group_data_dict[group_name]["supattr_conf_aggs"][supattr_name],
                    multiattr_metadata,
                    supattr_name,
                    group_name,
                    pn_normalized=plot_config["pn_normalized"],
                    plot_filename=plot_filename
                )

    # Category confusion metric plots

    for plot_config in ana_plots_settings["multiattr_conf_metric"]:

        for group_name, metric_name in itertools.product(
            plot_config["group_names"],
            plot_config["metric_names"]
        ):
            
            for supattr_name in multiattr_metadata.supattr_name_list:

                plot_name = group_name + "___" + metric_name + "___" + supattr_name
                if plot_config["pn_normalized"]: plot_name += "___pn_norm"
                if plot_config["show_numbers"]: plot_name += "___nums"
                plot_filename = os.path.join(multiattr_conf_metric_plots_dirname, plot_name + ".jpg")

                plot_supattr_conf_metric(
                    group_data_dict[group_name]["supattr_conf_aggs"][supattr_name],
                    metric_fun_dict[metric_name],
                    multiattr_metadata,
                    supattr_name,
                    group_name,
                    metric_name,
                    pn_normalized=plot_config["pn_normalized"],
                    show_numbers=plot_config["show_numbers"],
                    plot_filename=plot_filename
                )
