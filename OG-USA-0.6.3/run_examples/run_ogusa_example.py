'''
Example script for setting policy and running OG-USA.
'''

# import modules
import multiprocessing
from distributed import Client
import time
import numpy as np
import os
import taxcalc
from taxcalc import Calculator
from ogusa import output_tables as ot
from ogusa import output_plots as op
from ogusa.execute import runner
from ogusa.constants import REFORM_DIR, BASELINE_DIR
from ogusa.utils import safe_read_pickle


def main():
    # Define parameters to use for multiprocessing
    client = Client()
    num_workers = min(multiprocessing.cpu_count(), 7)
    print('Number of workers = ', num_workers)
    run_start_time = time.time()

    # Grab a reform JSON file already in Tax-Calculator
    # In this example the 'reform' is a change to 2017 law (the
    # baseline policy is tax law in 2018)
    reform_url = ('https://raw.githubusercontent.com/'
                  'PSLmodels/Tax-Calculator/' + taxcalc.__version__ +
                  '/taxcalc/reforms/2017_law.json')
    ref = Calculator.read_json_param_objects(reform_url, None)
    iit_reform = ref['policy']

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, BASELINE_DIR)
    reform_dir = os.path.join(CUR_DIR, REFORM_DIR)

    # Set some OG model parameters
    # See default_parameters.json for more description of these parameters
    alpha_T = np.zeros(50)  # Adjusting the path of transfer spending
    alpha_T[0:2] = 0.09
    alpha_T[2:10] = 0.09 + 0.01
    alpha_T[10:40] = 0.09 - 0.01
    alpha_T[40:] = 0.09
    alpha_G = np.zeros(7)  # Adjusting the path of non-transfer spending
    alpha_G[0:3] = 0.05 - 0.01
    alpha_G[3:6] = 0.05 - 0.005
    alpha_G[6:] = 0.05
    # Set start year for baseline and reform.
    START_YEAR = 2021
    # Also adjust the Frisch elasticity, the start year, the
    # effective corporate income tax rate, and the SS debt-to-GDP ratio
    og_spec = {'frisch': 0.41, 'start_year': START_YEAR, 'cit_rate': [0.21],
               'debt_ratio_ss': 1.0, 'alpha_T': alpha_T.tolist(),
               'alpha_G': alpha_G.tolist()}

    '''
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    '''
    tax_func_path = os.path.join(
        CUR_DIR, '..', 'ogusa', 'data', 'tax_functions',
        'TxFuncEst_baseline_CPS.pkl')  # use cached baseline estimates
    kwargs = {'output_base': base_dir, 'baseline_dir': base_dir,
              'test': False, 'time_path': True, 'baseline': True,
              'og_spec': og_spec, 'guid': '_example',
              'run_micro': False, 'tax_func_path': tax_func_path,
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
    og_spec = {'frisch': 0.41, 'start_year': START_YEAR,
               'cit_rate': [0.35], 'debt_ratio_ss': 1.0,
               'alpha_T': alpha_T.tolist(),
               'alpha_G': alpha_G.tolist()}
    kwargs = {'output_base': reform_dir, 'baseline_dir': base_dir,
              'test': False, 'time_path': True, 'baseline': False,
              'og_spec': og_spec, 'guid': '_example',
              'iit_reform': iit_reform, 'run_micro': True, 'data': 'cps',
              'client': client, 'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    # return ans - the percentage changes in macro aggregates and prices
    # due to policy changes from the baseline to the reform
    base_tpi = safe_read_pickle(
        os.path.join(base_dir, 'TPI', 'TPI_vars.pkl'))
    base_params = safe_read_pickle(
        os.path.join(base_dir, 'model_params.pkl'))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, 'TPI', 'TPI_vars.pkl'))
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, 'model_params.pkl'))
    ans = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['Y', 'C', 'K', 'L', 'r', 'w'], output_type='pct_diff',
        num_years=10, start_year=og_spec['start_year'])

    # create plots of output
    op.plot_all(base_dir, reform_dir,
                os.path.join(CUR_DIR, 'run_example_plots'))

    print("total time was ", (time.time() - run_start_time))
    print('Percentage changes in aggregates:', ans)
    # save percentage change output to csv file
    ans.to_csv('ogusa_example_output.csv')
    client.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
