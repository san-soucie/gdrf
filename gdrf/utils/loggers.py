
"""
Logging utils
Adapted from YOLOv5, https://github.com/ultralytics/yolov5/
"""

import warnings
from threading import Thread

import torch

from .general import colorstr, emojis
from .wandblogger import WandbLogger

LOGGERS = ('csv', 'wandb')  # text-file, Weights & Biases

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None




class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = ['train/loss',  # train loss
                     'metrics/perplexity',   # metrics
                     'x/kernel.lengthscale', 'x/kernel.variance']  # params
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            print(emojis(s))

        # TensorBoard
        s = self.save_dir

        # W&B
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        pass

    def on_train_batch_end(self, *args, **kwargs):
        # Callback runs on train batch end
        if self.wandb and kwargs.get('log', None) is not None:
            self.wandb.log(kwargs['log'])

    def on_train_epoch_end(self, *args, **kwargs):
        # Callback runs on train epoch end
        if self.wandb:
            current_epoch = kwargs.get('epoch', self.wandb.current_epoch)
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
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, plots, epoch):
        # Callback runs on training end
        # if plots:
        #     files = plot_results(file=self.save_dir / 'results.csv')  # save results.png
        # files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter


        if self.wandb:
            # self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            if not self.opt.evolve:
                self.wandb.finish_run(artifact_or_path=str(best if best.exists() else last), type='model',
                                   name='run_' + self.wandb.wandb_run.id + '_model',
                                   aliases=['latest', 'best', 'stripped'])
            else:
                self.wandb.finish_run()
                self.wandb = WandbLogger(self.opt)
