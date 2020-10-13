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


    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, BASELINE_DIR)
    reform_dir = os.path.join(CUR_DIR, REFORM_DIR)


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

    # ETR
    ans6 = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['ETRss'], output_type='levels',
        num_years=10, start_year=2021)

    # create plots of output
    op.plot_all(base_dir, reform_dir,
                os.path.join(CUR_DIR, 'run_biden_plots'))

    # save output tables to csv files
    ans0.to_csv('ogusa_biden_output_2021-30.csv')
    ans1.to_csv('ogusa_biden_output_2031-40.csv')
    ans2.to_csv('ogusa_biden_dynamic_revenue.csv')
    ans3.to_csv('ogusa_biden_macro_table.csv')
    ans4.to_csv('ogusa_biden_debt2gdp_2021-30.csv')
    ans5.to_csv('ogusa_biden_output_dump.csv')
    ans6.to_csv('ogusa_biden_etr.csv')


if __name__ == "__main__":
    # execute only if run as a script
    main()
