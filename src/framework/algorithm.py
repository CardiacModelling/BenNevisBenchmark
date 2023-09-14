from .algorithm_instance import AlgorithmInstance
from .config import RS_ITER_NUM, MAX_INSTANCE_FES, logger
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import pandas as pd
# import seaborn
import optuna
import os


class Algorithm:
    def __init__(self, 
                 name: str,
                 func, 
                 version: int=1):
        """
        Class for an algorithm.

        Parameters
        ----------
        name : string
            The name of the algorithm.
        func : function
            The function that runs the algorithm. This function takes all the
            hyper-paramters used by the algorithm as keyword arguments, and
            returns a ``Result`` object.
        version : int
            The version of the algorithm. This is used to distinguish between
            different versions of the same algorithm so that they are saved in
            different folders.
        """
        self.name = name
        self.func = func
        self.version = version

        self.best_instance = None
    
    @property
    def info(self):
        return {
            'algorithm_name': self.name,
            'algorithm_version': self.version,
        }


    def tune_params(
        self,
        db_path,
        iter_num=RS_ITER_NUM,
        measure='gary_ert',
        direction='minimize',
        save_handler=None,
        max_instance_fes=MAX_INSTANCE_FES,
        plot_path=None,
        make_all_plots=True,
    ):
        
        def objective(trial: optuna.Trial):
            instance = AlgorithmInstance(self, trial)
            instance.run(save_handler, max_instance_fes)
            if make_all_plots and plot_path is not None:
                instance.plot_convergence_graph(img_path=f'{plot_path}/{trial._trial_id}-c.png')
                instance.plot_stacked_graph(img_path=f'{plot_path}/{trial._trial_id}-s.png')
            return instance.performance_measures()[measure]
        
        study = optuna.create_study(
            direction=direction, 
            study_name=f'{self.name}-{self.version}',
            storage=f'sqlite:///{db_path}',
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=iter_num)
        self.best_instance = AlgorithmInstance(self, study.best_trial)

        if not make_all_plots and plot_path is not None:
            self.best_instance.run(
                restart=True, 
                save_handler=save_handler,
                save_partial=False,
                does_prune=False,
            )
            best_id = study.best_trial._trial_id
            self.best_instance.plot_convergence_graph(img_path=f'{plot_path}/best-{best_id}-c.png')
            self.best_instance.plot_stacked_graph(img_path=f'{plot_path}/best-{best_id}-s.png')
    
    def load_best_instance(self, db_path):
        study = optuna.load_study(
            study_name=f'{self.name}-{self.version}',
            storage=f'sqlite:///{db_path}',
        )
        self.best_instance = AlgorithmInstance(self, study.best_trial)