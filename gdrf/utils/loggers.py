"""
Logging utils
Adapted from YOLOv5, https://github.com/ultralytics/yolov5/
"""
from typing import List

import holoviews as hv
import pandas as pd
import torch

import gdrf.models

from ..visualize import matrix_plot, maxplot_2d, stackplot_1d
from .general import colorstr, emojis
from .wandblogger import WandbLogger

LOGGERS = ("csv", "wandb")  # text-file, Weights & Biases

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None


def _artifacts(
    save_dir,
    ckpt: str,
    index,
    obs_cats: List[str],
    xs: torch.Tensor,
    ws: torch.Tensor,
    visualize_results: bool = True,
):
    model = torch.load(ckpt, map_location=xs.device)["model"]
    ws_np = ws.detach().cpu().numpy()
    obs_df = pd.DataFrame(ws_np, columns=obs_cats, index=index)
    obs_df.to_csv(save_dir / "observations.csv")
    files = dict()
    if isinstance(model, gdrf.models.AbstractGDRF):
        n_dims = model.dims
        n_topics = model.K
        topic_probs = model.topic_probs(xs).detach().cpu().numpy()
        word_probs = model.word_probs(xs).detach().cpu().numpy()

        word_topic_matrix = model.word_topic_matrix.detach().cpu().numpy()
        topic_prob_df = pd.DataFrame(topic_probs, index=index)
        word_prob_df = pd.DataFrame(word_probs, index=index)
        word_topic_matrix_df = pd.DataFrame(
            word_topic_matrix,
            index=[f"topic {k}" for k in range(n_topics)],
            columns=obs_cats,
        )
        if visualize_results:
            plots = {
                "word_topic_matrix.png": matrix_plot(word_topic_matrix_df, log=False)
            }
            if n_dims == 1:
                plots["topic_prob.png"] = stackplot_1d(topic_prob_df, legend=True)
                plots["word_prob.png"] = stackplot_1d(word_prob_df)
                plots["observations.png"] = stackplot_1d(obs_df)
            elif n_dims == 2:
                plots["topic_prob.png"] = maxplot_2d(topic_prob_df)
                plots["word_prob.png"] = maxplot_2d(word_prob_df)
                plots["observations.png"] = maxplot_2d(obs_df)
            for plot_name, plot in plots.items():
                hv.save(plot, save_dir / plot_name)
                files[f"Results/{plot_name}"] = wandb.Image(
                    str(save_dir / plot_name), caption=plot_name
                )
        topic_prob_df.to_csv(save_dir / "topic_probs.csv")
        word_prob_df.to_csv(save_dir / "word_probs.csv")
        word_topic_matrix_df.to_csv(save_dir / "word_topic_matrix.csv")
        # files["Results/observations.csv"] = wandb.Table(data=obs_df)
        # files["Results/topic_probs.csv"] = wandb.Table(dataframe=topic_prob_df)
        # files["Results/word_probs.csv"] = wandb.Table(dataframe=word_prob_df)
        # files["Results/word_topic_matrix.csv"] = wandb.Table(dataframe=word_topic_matrix_df)
    return files


class Loggers:
    # YOLOv5 Loggers class
    def __init__(
        self,
        save_dir=None,
        weights=None,
        opt=None,
        logger=None,
        include=LOGGERS,
    ):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "train/loss",  # train loss
            "metrics/perplexity",  # metrics
            "x/kernel.lengthscale",
            "x/kernel.variance",
        ]  # params
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr("Weights & Biases: ")
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            print(emojis(s))

        # TensorBoard
        s = self.save_dir

        # W&B
        if wandb and "wandb" in self.include:
            wandb_artifact_resume = isinstance(
                self.opt["resume"], str
            ) and self.opt.resume.startswith("wandb-artifact://")
            run_id = (
                torch.load(self.weights).get("wandb_id")
                if self.opt["resume"] and not wandb_artifact_resume
                else None
            )
            self.wandb = WandbLogger(self.opt, run_id=run_id)
            self.opt = wandb.config
        else:
            self.wandb = None

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        pass

    def on_train_batch_end(self, *args, **kwargs):
        # Callback runs on train batch end
        if self.wandb and kwargs.get("log", None) is not None:
            self.wandb.log(kwargs["log"])

    def on_train_epoch_end(self, *args, **kwargs):
        # Callback runs on train epoch end
        if self.wandb:
            current_epoch = kwargs.get("epoch", self.wandb.current_epoch)
            self.wandb.current_epoch = current_epoch + 1

    def on_val_image_end(self, *args, **kwargs):
        # Callback runs on val image end
        pass

    def on_val_end(self, *args, **kwargs):
        # Callback runs on val end
        pass

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv:
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = (
                ""
                if file.exists()
                else (("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")
            )  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            if (
                (epoch + 1) % self.opt["save_period"] == 0 and not final_epoch
            ) and self.opt["save_period"] != -1:
                self.wandb.log_model(
                    last.parent, self.opt, epoch, fi, best_model=best_fitness == fi
                )

    def on_train_end(
        self, last, best, xs, ws, index, obs_cats, epoch, visualize_results
    ):
        model_ckpt = str(best if best.exists() else last)
        results = _artifacts(
            self.save_dir, model_ckpt, index, obs_cats, xs, ws, visualize_results
        )

        # Callback runs on training end
        # if plots:
        #     files = plot_results(file=self.save_dir / 'results.csv')  # save results.png
        # files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter

        # Save training data
        # Save topics at training locations
        # Save word-topic matrix
        # Save images of the above

        if self.wandb:
            self.wandb.log(results)
            self.wandb.finish_run(
                artifact_or_path=model_ckpt,
                type="model",
                name="run_" + self.wandb.wandb_run.id + "_model",
                aliases=["latest", "best", "stripped"],
            )
