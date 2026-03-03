import argparse
import os
import random
import sys
import gc
from collections import defaultdict
from itertools import chain
from pathlib import Path

import hydra
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import colors as mcolors
from huggingface_hub import hf_hub_download, login
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from omegaconf import DictConfig

from tools.run_infinity import *
from steering.activation_capture import (
    save_infinity_activations,
    get_t5_activations,
)
from utils.datasets_utils import split_dataset_by_categories


def config_infinity(infinity_config: dict) -> argparse.Namespace:
    sys.path.insert(0, os.getcwd())
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # model_path = f"Infinity/weights/{infinity_config.MODEL_NAME}"
    # vae_path = f"Infinity/weights/{infinity_config.VAE_NAME}"
    model_path = f"weights/{infinity_config.MODEL_NAME}"
    vae_path = f"weights/{infinity_config.VAE_NAME}"
    text_encoder_ckpt = infinity_config.TEXT_ENCODER_CKPT
    args = argparse.Namespace(
        pn=infinity_config.PIXEL_NUMBER,
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=infinity_config.VAE_TYPE,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=True,
        model_type=infinity_config.MODEL_TYPE,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir="./Infinity/tmp",
        checkpoint_type="torch",
        seed=0,
        bf16=1,
        save_file="tmp.jpg",
        enable_model_cache=False,
    )
    return args


def load_infinity_weights(args, infinity_config):
    os.chdir("Infinity")
    login(token=infinity_config.HF_HUB_TOKEN)

    hf_hub_download(
        repo_id=infinity_config.REPOSITORY,
        filename=infinity_config.VAE_NAME,
        local_dir="weights/",
    )
    hf_hub_download(
        repo_id=infinity_config.REPOSITORY,
        filename=infinity_config.MODEL_NAME,
        local_dir="weights/",
    )
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    return infinity, vae, text_encoder, text_tokenizer


def setup_infinity(infinity, vae, text_tokenizer, text_encoder, args, cfg):
    h_div_w = 1 / 1
    h_div_w_template_ = h_div_w_templates[
        np.argmin(np.abs(h_div_w_templates - h_div_w))
    ]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    tau = 0.5
    seed = random.randint(0, 10000)

    return infinity, vae, text_tokenizer, text_encoder, scale_schedule, cfg, tau, seed


def _iter_loc_batches(paths):
    for path in paths:
        activations = torch.load(path, map_location="cpu")
        for loc, acts in activations.items():
            safe_act = acts[::2]
            unsafe_act = acts[1::2]
            yield loc, safe_act, unsafe_act
        del activations
        gc.collect()


def analyze_representation_differences(cache_dir, train_split=0.6, seed=42, alpha=1e-3):
    ext = "*.pt"
    paths_iterables = [cache_dir.rglob(ext)]
    paths = sorted(p for p in chain.from_iterable(paths_iterables) if p.is_file())

    sum_safe = {}
    sum_unsafe = {}
    counts = defaultdict(int)

    scalers = {}

    rng = np.random.RandomState(seed)
    for loc, safe_act, unsafe_act in _iter_loc_batches(paths):
        if loc not in sum_safe:
            sum_safe[loc] = torch.zeros_like(safe_act[0])
            sum_unsafe[loc] = torch.zeros_like(unsafe_act[0])
        sum_safe[loc] += safe_act.sum(dim=0)
        sum_unsafe[loc] += unsafe_act.sum(dim=0)
        counts[loc] += safe_act.shape[0]

        B = safe_act.shape[0]
        mask_train = rng.rand(B) < train_split
        if mask_train.sum() == 0:
            continue
        X_train = torch.cat(
            [safe_act[mask_train], unsafe_act[mask_train]], dim=0
        ).numpy()
        scaler = scalers.get(loc)
        if scaler is None:
            scaler = scalers[loc] = StandardScaler()
        scaler.partial_fit(X_train)
    gc.collect()

    clfs = fit_classifier(paths, scalers, train_split, alpha, seed)

    y_true, y_score = evaluate_classifier(paths, scalers, clfs, train_split, seed)

    return calculate_stats(sum_safe, sum_unsafe, counts, y_true, y_score)


def fit_classifier(paths, scalers, train_split, alpha, seed):
    rng = np.random.RandomState(seed)
    clfs = {}
    for loc, safe_act, unsafe_act in _iter_loc_batches(paths):
        B = safe_act.shape[0]
        mask_train = rng.rand(B) < train_split
        if mask_train.sum() == 0:
            continue

        safe_train = safe_act[mask_train]
        unsafe_train = unsafe_act[mask_train]

        X_train = torch.cat([safe_train, unsafe_train], dim=0).numpy()
        y_train = np.concatenate(
            [
                np.zeros(safe_train.shape[0], dtype=np.int64),
                np.ones(unsafe_train.shape[0], dtype=np.int64),
            ]
        )

        X_train = scalers[loc].transform(X_train)

        clf = clfs.get(loc)
        if clf is None:
            clf = clfs[loc] = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=1,
                learning_rate="optimal",
                random_state=seed,
            )
            clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        else:
            clf.partial_fit(X_train, y_train)
    gc.collect()
    return clfs


def evaluate_classifier(paths, scalers, clfs, train_split, seed):
    y_true = defaultdict(list)
    y_score = defaultdict(list)
    rng = np.random.RandomState(seed)
    for loc, safe_act, unsafe_act in _iter_loc_batches(paths):
        B = safe_act.shape[0]
        mask_train = rng.rand(B) < train_split
        mask_test = ~mask_train
        if mask_test.sum() == 0:
            continue

        safe_test = safe_act[mask_test]
        unsafe_test = unsafe_act[mask_test]

        X_test = torch.cat([safe_test, unsafe_test], dim=0).numpy()
        y_test = np.concatenate(
            [
                np.zeros(safe_test.shape[0], dtype=np.int64),
                np.ones(unsafe_test.shape[0], dtype=np.int64),
            ]
        )

        X_test = scalers[loc].transform(X_test)
        probs = clfs[loc].predict_proba(X_test)[:, 1]

        y_true[loc].append(y_test)
        y_score[loc].append(probs)
    gc.collect()
    return y_true, y_score


def calculate_stats(sum_safe, sum_unsafe, counts, y_true, y_score):
    results = defaultdict(list)
    locs = sorted(sum_safe.keys())
    for scale, layer_id in locs:
        loc = (scale, layer_id)

        mu_s = sum_safe[loc] / counts[loc]
        mu_u = sum_unsafe[loc] / counts[loc]
        dim = mu_s.shape[0]

        l2 = torch.norm(mu_u - mu_s).item() / np.sqrt(dim)
        cos_dis = 1 - F.cosine_similarity(mu_s.unsqueeze(0), mu_u.unsqueeze(0)).item()

        y_loc = np.concatenate(y_true[loc])
        p_loc = np.concatenate(y_score[loc])
        auc = roc_auc_score(y_loc, p_loc)

        results["scale"].append(scale)
        results["layer"].append(layer_id)
        results["L2_difference"].append(round(l2, 4))
        results["Cosine distance"].append(round(cos_dis, 4))
        results["Linear probe AUC"].append(round(auc, 4))
    return results


def plot_heatmap(
    results, category, value, ax, annot=True, annot_size=8, max=None, min=None
):
    heat_auc = results.pivot(index="scale", columns="layer", values=value)

    sns.heatmap(
        data=heat_auc,
        annot=annot,
        annot_kws={"size": annot_size},
        fmt=".3f",
        cmap="viridis",
        vmin=min,
        vmax=max,
        ax=ax,
    )
    ax.set_title(f"Category: {category}")


def plot_text_barplots(keys, results_text, category, category_dir):
    df = results_text.copy()

    df["layer"] = df["layer"].astype(int)
    df = df.sort_values("layer")

    for metric in keys:
        df[metric] = df[metric].astype(float)
        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(data=df, x="layer", y=metric, ax=ax)

        ax.set_title(f"Text Representation Differences - {category} - {metric}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)

        fig.tight_layout()
        fig.savefig(
            category_dir / f"text_{metric.replace(' ', '_')}_bar.png",
            dpi=300,
        )
        plt.close(fig)


def create_rank_table(results, metric, top_k=10):
    rank_table = (
        results[["scale", "layer", metric]]
        .sort_values(by=metric, ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    rank_table[metric] = rank_table[metric].round(4)
    rank_table["layer"] = rank_table["layer"].astype(int).astype(str)
    rank_table["scale"] = rank_table["scale"].astype(int).astype(str)
    return rank_table


def create_means_table(results, index, metric):
    means_table = results.groupby(index)[metric].mean().reset_index()
    means_table[metric] = means_table[metric].round(4)
    means_table[index] = means_table[index].astype(int).astype(str)

    return means_table


def draw_highlighted_means_table(ax, means_table, metric, title, top_k=3):
    df = means_table.reset_index(drop=True).copy()
    n_rows, n_cols = df.shape

    top_idx = df[metric].nlargest(top_k).index.tolist()

    default = (1.0, 1.0, 1.0, 1.0)
    highlight = mcolors.to_rgba("#ffe9a6")

    cell_colours = [[default for _ in range(n_cols)] for _ in range(n_rows)]
    for i in top_idx:
        for j in range(n_cols):
            cell_colours[i][j] = highlight

    idx_col = df.columns[0]
    df[idx_col] = df[idx_col].astype(str)
    df[metric] = df[metric].map("{:.4f}".format)

    ax.axis("off")
    ax.set_title(title, fontsize=16)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns.tolist(),
        cellLoc="center",
        loc="center",
        cellColours=cell_colours,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.6)

    return table


def setup_grid_plots(keys):
    metric_layout = dict()

    for key in keys:
        fig = plt.figure(figsize=(22, 26), dpi=300)
        outer = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.6, 5.4])

        top_gs = outer[0].subgridspec(1, 1)

        tables_gs = outer[1].subgridspec(1, 3)

        bottom_gs = outer[2].subgridspec(3, 2)

        metric_layout[key] = dict(
            fig=fig,
            top_gs=top_gs,
            tables_gs=tables_gs,
            bottom_gs=bottom_gs,
        )
    return metric_layout


def plot_tables(results_visual, metric, fig, tables_gs):
    top_k = 10
    rank_table = create_rank_table(results_visual, metric, top_k=top_k)
    scale_means_table = create_means_table(results_visual, "scale", metric)
    layer_means_table = create_means_table(results_visual, "layer", metric)

    rank_table_to_plot = rank_table
    gs_to_use = tables_gs[0, 0]
    title = f"Top {top_k} representation differences"
    ax_table = fig.add_subplot(gs_to_use)
    ax_table.axis("off")
    ax_table.set_title(title, fontsize=16)
    table = ax_table.table(
        cellText=rank_table_to_plot.values,
        colLabels=rank_table_to_plot.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.6)

    ax_scale = fig.add_subplot(tables_gs[0, 1])
    draw_highlighted_means_table(
        ax_scale,
        scale_means_table,
        metric,
        title="Mean representation differences by scale",
        top_k=3,
    )

    ax_layer = fig.add_subplot(tables_gs[0, 2])
    draw_highlighted_means_table(
        ax_layer,
        layer_means_table,
        metric,
        title="Mean representation differences by layer",
        top_k=3,
    )


def plot_grids(
    keys, metric_layout, results_visual, category, category_index, metric_limits
):
    for metric in keys:
        layout = metric_layout[metric]
        fig = layout["fig"]
        top_gs = layout["top_gs"]
        tables_gs = layout["tables_gs"]
        bottom_gs = layout["bottom_gs"]
        if category == "All":
            ax_big = fig.add_subplot(top_gs[0, 0])
            plot_heatmap(
                results_visual,
                category,
                metric,
                ax_big,
                annot_size=9,
                min=metric_limits[metric][0],
                max=metric_limits[metric][1],
            )
            plot_tables(results_visual, metric, fig, tables_gs)
        else:
            small_idx = category_index - 1
            row = small_idx // 2
            col = small_idx % 2
            ax_small = fig.add_subplot(bottom_gs[row, col])
            plot_heatmap(
                results_visual,
                category,
                metric,
                ax=ax_small,
                annot_size=8,
                min=metric_limits[metric][0],
                max=metric_limits[metric][1],
            )


def save_grid_plots(keys, metric_layout, save_root):
    for metric in keys:
        fig = metric_layout[metric]["fig"]
        fig.suptitle(
            f"Vision Representation Differences - {metric}",
            fontsize=20,
            y=0.99,
        )
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.96,
            bottom=0.04,
            left=0.04,
            right=0.99,
            hspace=0.4,
            wspace=0.3,
        )
        fig.savefig(save_root / f"{metric.replace(' ', '_')}_full_panel.png")
        plt.close(fig)


def get_activations(cache_dir, category, extract_activ_func, dataset, **kwargs):
    if not any(cache_dir.iterdir()):
        print(f"[{category}] computing activations...")
        extract_activ_func(dataset[0], **kwargs)
    else:
        print(f"[{category}] activations cached on drive...")


@hydra.main(config_path="config", config_name="representation", version_base="1.2")
def main(cfg: DictConfig):
    torch.set_grad_enabled(False)

    experiment_config = cfg.experiment
    infinity_config = cfg.infinity
    print("[Loading dataset]")
    datasets = split_dataset_by_categories(config=experiment_config)
    print("<Finished loading dataset>")

    infinity_args = config_infinity(infinity_config)

    infinity, vae, text_encoder, text_tokenizer = load_infinity_weights(
        infinity_args, infinity_config
    )
    infinity_setup = setup_infinity(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        infinity_args,
        infinity_config.CFG,
    )
    save_root = Path(cfg.save_folder)
    keys = ["L2_difference", "Cosine distance", "Linear probe AUC"]
    metric_limits = {
        "L2_difference": (cfg.l2_bounds[0], cfg.l2_bounds[1]),
        "Cosine distance": (cfg.cos_dist_bounds[0], cfg.cos_dist_bounds[1]),
        "Linear probe AUC": (cfg.lin_prb_auc_bounds[0], cfg.lin_prb_auc_bounds[1]),
    }
    metric_layout = setup_grid_plots(keys)

    for category_index, (category, dataset) in enumerate(datasets.items()):
        category_dir = save_root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        text_cache_dir = category_dir / Path("text")
        text_cache_dir.mkdir(parents=True, exist_ok=True)
        vis_cache_dir = category_dir / Path("vis")
        vis_cache_dir.mkdir(parents=True, exist_ok=True)

        if "text" in cfg.modalities:
            get_activations(
                text_cache_dir,
                category,
                get_t5_activations,
                dataset,
                tokenizer=text_tokenizer,
                encoder=text_encoder,
                layers_to_capture=range(1, 24),
                safe_class=experiment_config.SAFE_CLASS,
                batch_size=experiment_config.BATCH_SIZE,
            )
            print(f"[Text]")
            results_text = pd.DataFrame(
                analyze_representation_differences(text_cache_dir)
            )
            results_text.to_csv(
                category_dir / "text_representation_differences.csv", index=False
            )
            plot_text_barplots(keys, results_text, category, category_dir)
        if "vision" in cfg.modalities:
            infinity_setup[0].configure_activation_capture()
            get_activations(
                vis_cache_dir,
                category,
                save_infinity_activations,
                dataset,
                batch_size=experiment_config.BATCH_SIZE,
                infinity_setup=infinity_setup,
                vae_type=infinity_config.VAE_TYPE,
                cache_path=vis_cache_dir,
            )
            infinity_setup[0].reset_activation_capture()
            torch.cuda.empty_cache()
            print(f"[Vision]")
            results_visual = pd.DataFrame(
                analyze_representation_differences(vis_cache_dir)
            )
            plot_grids(
                keys,
                metric_layout,
                results_visual,
                category,
                category_index,
                metric_limits,
            )
            results_visual.to_csv(
                category_dir / "vision_representation_differences.csv", index=False
            )

    save_grid_plots(
        keys,
        metric_layout,
        save_root,
    )


if __name__ == "__main__":
    main()
