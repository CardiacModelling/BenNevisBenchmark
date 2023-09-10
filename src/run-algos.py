from algorithms import cmaes, mlsl, dual_annealing
from framework import *
import click

algo_list = [cmaes, mlsl, dual_annealing]

@click.command()
@click.argument('algo_name')
@click.option('--db_name', default=None, help='Specify the database name')
@click.option('--iter_num', default=RS_ITER_NUM, type=int, help='Specify the iteration number in hyper-parameter random search')
@click.option('--max_instance_fes_ht', default=MAX_INSTANCE_FES, type=int, help='Specify max instance fes for hyper-parameter tuning')
@click.option('--max_instance_fes_best', default=MAX_INSTANCE_FES * 5, type=int, help='Specify max instance fes for the best instance selected')
def run_algo(
    algo_name,
    db_name=None,
    iter_num=RS_ITER_NUM,
    max_instance_fes_ht=MAX_INSTANCE_FES,
    max_instance_fes_best=MAX_INSTANCE_FES * 5,
    save_type='json',
):
    algo = None
    for a in algo_list:
        if a.name == algo_name:
            algo = a
            break
    else:
        raise ValueError('Unknown algorithm name provided.')
    
    if db_name is None:
        db_name = algo_name
    
    if save_type == 'json':
        save_handler = SaveHandlerJSON(db_name, db_name)
    elif save_type == 'mongo':
        save_handler = SaveHandlerMongo(db_name)
    
    algo.load_best_instance(save_handler=save_handler)
    algo.load_instance_indices(save_handler=save_handler)
    algo.tune_params(
        save_handler=save_handler,
        iter_num=iter_num, 
        max_instance_fes=max_instance_fes_ht,
    )
    ins = algo.best_instance
    ins.run(
        restart=True, 
        max_instance_fes=max_instance_fes_best,
        save_handler=save_handler,
        save_partial=False,
    )
    ins.plot_convergence_graph(img_path=f'{algo_name}-convergence.png')
    ins.plot_stacked_graph(img_path=f'{algo_name}-stacked-last.png')
    ins.plot_stacked_graph(img_path=f'{algo_name}-stacked-judge.png', mode='judge')
    ins.plot_stacked_graph(img_path=f'{algo_name}-stacked-terminate.png', mode='terminate')
    click.echo(ins.info)
    click.echo(ins.params)
    click.echo(ins.performance_measures(excluding_first=True))

if __name__ == '__main__':
    algo_names = [algo.name for algo in algo_list]
    click.echo('List of selectable algorithms:')
    click.echo(algo_names)
    run_algo()