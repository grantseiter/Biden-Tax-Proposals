'''
Run of OG-USA to simulate the Biden plan
'''

# import modules
import multiprocessing
from distributed import Client
import time
import numpy as np
import os
from taxcalc import Calculator
from ogusa import output_tables as ot
from ogusa import output_plots as op
from ogusa.execute import runner
from ogusa.constants import REFORM_DIR, BASELINE_DIR
from ogusa.utils import safe_read_pickle


def main():
    # Define parameters to use for multiprocessing
    client = Client()
    num_workers = min(multiprocessing.cpu_count(), 12)
    print('Number of workers = ', num_workers)
    run_start_time = time.time()

    reform_url = ('biden-iitax-reforms.json')
    ref = Calculator.read_json_param_objects(reform_url, None)
    iit_reform = ref['policy']

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, BASELINE_DIR)
    reform_dir = os.path.join(CUR_DIR, REFORM_DIR)


    '''
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    '''
    og_spec_base = {'start_year': 2021, 'tG1': 20, 'tG2': 200, 
                    'initial_debt_ratio': 0.982, 'debt_ratio_ss': 1.5,
                    'tax_func_type': 'DEP', 'age_specific': True}
    kwargs = {'output_base': base_dir, 'baseline_dir': base_dir,
              'test': False, 'time_path': True, 'baseline': True,
              'og_spec': og_spec_base, 'guid': '_biden',
              'run_micro': True, 'tax_func_path': None,
              'data': 'cps', 'client': client,
              'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    '''
    ------------------------------------------------------------------------
    Run reform policy
    ------------------------------------------------------------------------
    '''
    # update the effective corporate income tax rate
    og_spec_reform = {'start_year': 2021, 'tG1': 20, 'tG2': 200,
                      'initial_debt_ratio': 0.982, 'debt_ratio_ss': 1.5,
                      'tax_func_type': 'DEP', 'age_specific': True,
                      'cit_rate': [0.28]}
    kwargs = {'output_base': reform_dir, 'baseline_dir': base_dir,
              'test': False, 'time_path': True, 'baseline': False,
              'og_spec': og_spec_reform, 'guid': '_biden',
              'iit_reform': iit_reform, 'run_micro': True, 
              'tax_func_path': None, 'data': 'cps',
              'client': client, 'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    # return ans - the percentage changes in macro aggregates and prices
    # due to policy changes from the baseline to the reform
    base_tpi = safe_read_pickle(
        os.path.join(base_dir, 'TPI', 'TPI_vars.pkl'))
    base_ss = safe_read_pickle(
        os.path.join(base_dir, 'SS', 'SS_vars.pkl'))
    base_params = safe_read_pickle(
        os.path.join(base_dir, 'model_params.pkl'))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, 'TPI', 'TPI_vars.pkl'))
    reform_ss = safe_read_pickle(
        os.path.join(reform_dir, 'SS', 'SS_vars.pkl'))
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, 'model_params.pkl'))

    # Macro Agg Tables
    ans0 = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['Y', 'C', 'K', 'L', 'r', 'w'], output_type='pct_diff',
        num_years=10, start_year=2021)
    ans1 = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['Y', 'C', 'K', 'L', 'r', 'w'], output_type='pct_diff',
        num_years=10, start_year=2031)

    # Dynamic Revenue Feedback Table
    ans2 = ot.dynamic_revenue_decomposition(
        base_params, base_tpi, base_ss, reform_params, reform_tpi, reform_ss,
        num_years = 10, include_SS=False, include_overall=True,
        start_year=2021, table_format=None, path=None)

    # Data for Macro Aggregates Graph 
    ans3 = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['Y', 'K', 'L', 'C'], output_type='pct_diff',
        num_years=60, start_year=2021)

    # Data for Debt to GDP Graph
    ans4 = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['D','Y'], output_type='levels',
        num_years=60, start_year=2021)

    # Data Dump
    ans5 = ot.tp_output_dump_table(
        base_params, base_tpi, reform_params,
        reform_tpi, table_format=None, path=None)

    # create plots of output
    op.plot_all(base_dir, reform_dir,
                os.path.join(CUR_DIR, 'run_biden_plots'))

    print("total time was ", (time.time() - run_start_time))
    print('Percentage changes in aggregates:', ans)

    # save output tables to csv files
    ans0.to_csv('ogusa_biden_output_2021-30.csv')
    ans1.to_csv('ogusa_biden_output_2031-40.csv')
    ans2.to_csv('ogusa_biden_dynamic_revenue.csv')
    ans3.to_csv('ogusa_biden_macro_table.csv')
    ans4.to_csv('ogusa_biden_debt2gdp_2021-30.csv')
    ans5.to_csv('ogusa_biden_output_dump.csv')
    client.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
