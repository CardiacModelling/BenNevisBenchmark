from .algorithm_instance import AlgorithmInstance
from .config import RS_ITER_NUM, MAX_INSTANCE_FES
import optuna


class Algorithm:
    """
    Class for an algorithm, which is used to generate algorithm instances
    and tune the hyper-parameters of the algorithm. The `optuna` library is
    used for hyper-parameter tuning, and a ``Trial`` object is used to
    represent an algorithm instance.

    Attributes
    ----------
    name : string
        The name of the algorithm.
    func : function
        The function that runs the algorithm. This function should be defined
        using the ``optimizer`` decorator in ``runner.py``; alternatively, this function should
        take three arguments: ``rand_seed``, ``init_guess``, and ``trial``, where
        ``rand_seed`` is the random seed used for a run, ``init_guess`` is
        the initial guess for a run, and ``trial`` is the ``Trial`` object used for
        specifying the hyper-parameters of an algorithm instance.
    version : int
        The version of the algorithm. This is used to distinguish between
        different versions of the same algorithm when saving the results.
        When you change the implementation of an algorithm
        (i.e. change the function body of ``func``), you should
        change the version number.
    best_instance : AlgorithmInstance
        The best algorithm instance found by the hyper-parameter tuning process.

    Methods
    -------
    tune_params(db_path, iter_num, measure, direction, save_handler, max_instance_fes, plot_path, make_all_plots)
        Launch a hyper-parameter tuning process of the algorithm by generating ``iter_num``
        algorithm instances and saving the results to the specified database.
    load_best_instance(db_path)
        Load the best algorithm instance from the study database.
    """

    def __init__(self,
                 name: str,
                 func,
                 version: int = 1):
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
        """
        Tune the hyper-parameters of the algorithm.

        Parameters
        ----------
        db_path : string
            The path to the database file.
        iter_num : int
            The number of iterations to run the hyper-parameter tuning, i.e. the number of instances to generate.
        measure : string
            The performance measure to use for the hyper-parameter tuning.
        direction : 'minimize' or 'maximize'
            The direction to optimize the performance measure.
        save_handler : SaveHandler
            The save handler to use for saving the results.
        max_instance_fes : int
            The maximum number of function evaluations for a single instance.
        plot_path : string
            The path to save the plots.
        make_all_plots : bool
            Whether to make all plots or just the best instance's plots.
        """

        def objective(trial: optuna.Trial):
            instance = AlgorithmInstance(self, trial)
            instance.run(save_handler, max_instance_fes)
            if make_all_plots and plot_path is not None:
                instance.plot_convergence_graph(
                    img_path=f'{plot_path}/{trial._trial_id}-c.png')
                instance.plot_stacked_graph(
                    img_path=f'{plot_path}/{trial._trial_id}-s.png')
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
            self.best_instance.plot_convergence_graph(
                img_path=f'{plot_path}/best-{best_id}-c.png')
            self.best_instance.plot_stacked_graph(
                img_path=f'{plot_path}/best-{best_id}-s.png')

    def load_best_instance(self, db_path):
        study = optuna.load_study(
            study_name=f'{self.name}-{self.version}',
            storage=f'sqlite:///{db_path}',
        )
        self.best_instance = AlgorithmInstance(self, study.best_trial)
