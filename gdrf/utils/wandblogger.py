"""
Utilities and tools for tracking runs with Weights & Biases.
Adapted from YOLOv5, https://github.com/ultralytics/yolov5/
"""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .general import check_file, check_dataset

import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[3].as_posix())  # add yolov5/ to path

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'

def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]

def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    entity = run_path.parent.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return entity, project, run_id, model_artifact_name

def check_wandb_dataset(data_file):
    return {'train': data_file if data_file.startswith(WANDB_ARTIFACT_PREFIX) else check_dataset(data_file)}

def check_wandb_resume(resume):
    return isinstance(resume, str) and resume.startswith(WANDB_ARTIFACT_PREFIX)


class WandbLogger():
    """Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.
    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.
    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, opt_transforms = None, run_id=None, job_type='Training'):
        """
        - Initialize WandbLogger instance
        - Upload dataset if opt.upload_dataset is True
        - Setup trainig processes if job_type is 'Training'
        arguments:
        opt (namespace) -- Commandline arguments for this run
        opt_transforms (dict) -- Dictionary of transforms to apply when parsing config
        run_id (str) -- Run ID of W&B run to be resumed
        job_type (str) -- To set the job_type for this run
       """
        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run
        self.train_artifact = None
        self.train_artifact_path = None
        self.result_artifact = None
        self.max_imgs_to_log = 16
        self.wandb_artifact_data_dict = None
        self.data_dict = None
        self.opt_transforms = opt_transforms
        # It's more elegant to stick to 1 wandb.init call, but useful config data is overwritten in the WandbLogger's wandb.init call
        if isinstance(opt.resume, str):  # checks resume from artifact
            if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert wandb, 'install wandb to resume wandb runs'
                # Resume wandb-artifact:// runs here| workaround for not overwriting wandb.config
                self.wandb_run = wandb.init(id=run_id,
                                            project=project,
                                            entity=entity,
                                            resume='allow',
                                            allow_val_change=True)
                opt.resume = model_artifact_name
        elif self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project='gdrf' if opt.project == 'runs/train' else Path(opt.project).stem,
                                        entity=opt.entity,
                                        name=opt.name if opt.name != 'exp' else None,
                                        job_type=job_type,
                                        id=run_id,
                                        allow_val_change=True) if not wandb.run else wandb.run
        if self.wandb_run:
            if self.job_type == 'Training':
                if opt.upload_dataset:
                    if not opt.resume:
                        self.wandb_artifact_data_dict = self.check_and_upload_dataset(opt)

                if opt.resume:
                    # resume from artifact
                    if isinstance(opt.resume, str) and opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                        self.data_dict = dict(self.wandb_run.config.data_dict)
                    else:  # local resume
                        self.data_dict = check_wandb_dataset(opt.data)
                else:
                    self.data_dict = check_wandb_dataset(opt.data)
                    self.wandb_artifact_data_dict = self.wandb_artifact_data_dict or self.data_dict

                    # write data_dict to config. useful for resuming from artifacts. Do this only when not resuming.
                    self.wandb_run.config.update({'data_dict': self.wandb_artifact_data_dict},
                                                 allow_val_change=True)
                self.setup_training(opt)

            if self.job_type == 'Dataset Creation':
                self.data_dict = self.check_and_upload_dataset(opt)

    def check_and_upload_dataset(self, opt):
        """
        Check if the dataset format is compatible and upload it as W&B artifact
        arguments:
        opt (namespace)-- Commandline arguments for current run
        returns:
        Updated dataset info dictionary where local dataset paths are replaced by WAND_ARFACT_PREFIX links.
        """
        assert wandb, 'Install wandb to upload dataset'
        config_path = self.log_dataset_artifact(opt.data, 'GDRF' if opt.project == 'runs/train' else Path(opt.project).stem)
        print("Created dataset config file ", config_path)
        with open(config_path, errors='ignore') as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def setup_training(self, opt):
        """
        Setup the necessary processes for training YOLO models:
          - Attempt to download model checkpoint and dataset artifacts if opt.resume stats with WANDB_ARTIFACT_PREFIX
          - Update data_dict, to contain info of previous run if resumed and the paths of dataset artifact if downloaded
          - Setup log_dict, initialize bbox_interval
        arguments:
        opt (namespace) -- commandline arguments for this run
        """
        self.log_dict, self.current_epoch = {}, 0
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                config = self.wandb_run.config
                for k, v in config.items():
                    if k in opt:
                        if k in self.opt_transforms:
                            v = self.opt_transforms[k](v)
                        setattr(opt, k, v)

        data_dict = self.data_dict
        self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(data_dict.get('train'),
                                                                                       opt.artifact_alias)


        if self.train_artifact_path is not None:
            data_dict['train'] = str(Path(self.train_artifact_path))

        train_from_artifact = self.train_artifact_path is not None
        # Update the the data_dict to point to local artifacts dir
        if train_from_artifact:
            self.data_dict = data_dict

    def download_dataset_artifact(self, path, alias):
        """
        download the model checkpoint artifact if the path starts with WANDB_ARTIFACT_PREFIX
        arguments:
        path -- path of the dataset to be used for training
        alias (str)-- alias of the artifact to be download/used for training
        returns:
        (str, wandb.Artifact) -- path of the downladed dataset and it's corresponding artifact object if dataset
        is found otherwise returns (None, None)
        """
        if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
            artifact_path = Path(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + ":" + alias)
            dataset_artifact = wandb.use_artifact(artifact_path.as_posix().replace("\\", "/"))
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn\'t exist'"
            datadir = dataset_artifact.download()
            return datadir, dataset_artifact
        return None, None

    def download_model_artifact(self, opt):
        """
        download the model checkpoint artifact if the resume path starts with WANDB_ARTIFACT_PREFIX
        arguments:
        opt (namespace) -- Commandline arguments for this run
        """
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX) + ":latest")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            modeldir = model_artifact.download()
            epochs_trained = model_artifact.metadata.get('epochs_trained')
            total_epochs = model_artifact.metadata.get('total_epochs')
            is_finished = total_epochs is None
            assert not is_finished, 'training is finished, can only resume incomplete runs.'
            return modeldir, model_artifact
        return None, None

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as W&B artifact
        arguments:
        path (Path)   -- Path of directory containing the checkpoints
        opt (namespace) -- Command line arguments for this run
        epoch (int)  -- Current epoch number
        fitness_score (float) -- fitness score for current epoch
        best_model (boolean) -- Boolean representing if the current checkpoint is the best yet.
        """
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
            'original_url': str(path),
            'epochs_trained': epoch + 1,
            'save period': opt.save_period,
            'project': opt.project,
            'total_epochs': opt.epochs,
            'fitness_score': fitness_score
        })
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        wandb.log_artifact(model_artifact,
                           aliases=['latest', 'last', 'epoch ' + str(self.current_epoch), 'best' if best_model else ''])
        print("Saving model artifact on epoch ", epoch + 1)

    def log_dataset_artifact(self, data_file, single_cls, project, overwrite_config=False):
        """
        Log the dataset as W&B artifact and return the new data file with W&B links
        arguments:
        data_file (str) -- the .yaml file with information about the dataset like - path, classes etc.
        single_class (boolean)  -- train multi-class data as single-class
        project (str) -- project name. Used to construct the artifact path
        overwrite_config (boolean) -- overwrites the data.yaml file if set to true otherwise creates a new
        file with _wandb postfix. Eg -> data_wandb.yaml
        returns:
        the new .yaml file with artifact links. it can be used to start training directly from artifacts
        """
        self.data_dict = check_dataset(data_file)  # parse and check
        data = dict(self.data_dict)
        nc, names = (1, ['item']) if single_cls else (int(data['nc']), data['names'])
        names = {k: v for k, v in enumerate(names)}  # to index dictionary
        self.train_artifact = self.create_dataset_table(pd.read_csv(data['train'], index_col=0), name='train') if data.get('train') else None
        if data.get('train'):
            data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')
        if data.get('val'):
            data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')
        path = Path(data_file).stem
        path = (path if overwrite_config else path + '_wandb') + '.csv'  # updated data.yaml path
        data.pop('download', None)
        data.pop('path', None)

        if self.job_type == 'Training':  # builds correct artifact pipeline graph
            self.wandb_run.use_artifact(self.train_artifact)
        else:
            self.wandb_run.log_artifact(self.train_artifact)
        return path


    def create_dataset_table(self, dataset, name='dataset'):
        """
        Create and return W&B artifact containing W&B Table of the dataset.
        arguments:
        dataset (pandas.DataFrame) -- Dataframe with columns representing
        class_to_id (dict(int, str)) -- hash map that maps class ids to labels
        name (str) -- name of the artifact
        returns:
        dataset artifact to be logged or used
        """
        artifact = wandb.Artifact(name=name, type="dataset")
        table = wandb.Table(dataframe=dataset)
        artifact.add(table, name)
        return artifact

    def log(self, log_dict):
        """
        save the metrics to the logging dictionary
        arguments:
        log_dict (Dict) -- metrics/media to be logged in current step
        """
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        """
        commit the log_dict, model artifacts and Tables to W&B and flush the log_dict.
        arguments:
        best_result (boolean): Boolean representing if the result of this evaluation is best or not
        """
        if self.wandb_run:
            with all_logging_disabled():
                wandb.log(self.log_dict)
                self.log_dict = {}
            if self.result_artifact:
                wandb.log_artifact(self.result_artifact, aliases=['latest', 'last', 'epoch ' + str(self.current_epoch),
                                                                  ('best' if best_result else '')])
                self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")

    def finish_run(self, **kwargs):
        """
        Log metrics if any and finish the current W&B run
        """
        if all(kwargs.get(x, None) is not None for x in ['artifact_or_path', 'type', 'name', 'aliases']):
            wandb.log_artifact(
                artifact_or_path=kwargs.get('artifact_or_path'),
                type=kwargs.get('type'),
                name=kwargs.get('name'),
                aliases=kwargs.get('aliases')
            )
        if self.wandb_run:
            if self.log_dict:
                with all_logging_disabled():
                    wandb.log(self.log_dict)
            wandb.run.finish()


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """ source - https://gist.github.com/simon-weber/7853144
    A context manager that will prevent any logging messages triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL is defined.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)
