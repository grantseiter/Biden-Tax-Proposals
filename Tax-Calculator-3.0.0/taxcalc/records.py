"""
Tax-Calculator tax-filing-unit Records class.
"""
# CODING-STYLE CHECKS:
# pycodestyle records.py
# pylint --disable=locally-disabled records.py

import os
import numpy as np
import pandas as pd
from taxcalc.data import Data
from taxcalc.growfactors import GrowFactors
from taxcalc.utils import read_egg_csv


class Records(Data):
    """
    Records is a subclass of the abstract Data class, and therefore,
    inherits its methods (none of which are shown here).

    Constructor for the tax-filing-unit Records class.

    Parameters
    ----------
    data: string or Pandas DataFrame
        string describes CSV file in which records data reside;
        DataFrame already contains records data;
        default value is the string 'puf.csv'
        NOTE: when using custom data, set this argument to a DataFrame.
        NOTE: to use your own data for a specific year with Tax-Calculator,
        be sure to read the documentation on creating your own data file and
        then construct a Records object like this:
        mydata = pd.read_csv(<mydata.csv>)
        myrec = Records(data=mydata, start_year=<mydata_year>,
                        gfactors=None, weights=None)
        NOTE: data=None is allowed but the returned instance contains only
              the data variable information in the specified VARINFO file.

    start_year: integer
        specifies calendar year of the input data;
        default value is PUFCSV_YEAR.
        Note that if specifying your own data (see above NOTE) as being
        a custom data set, be sure to explicitly set start_year to the
        custom data's calendar year.

    gfactors: GrowFactors class instance or None
        containing record data growth (or extrapolation) factors.

    weights: string or Pandas DataFrame or None
        string describes CSV file in which weights reside;
        DataFrame already contains weights;
        None creates empty sample-weights DataFrame;
        default value is filename of the PUF weights.
        NOTE: when using custom weights, set this argument to a DataFrame.
        NOTE: assumes weights are integers that are 100 times the real weights.

    adjust_ratios: string or Pandas DataFrame or None
        string describes CSV file in which adjustment ratios reside;
        DataFrame already contains transposed/no-index adjustment ratios;
        None creates empty adjustment-ratios DataFrame;
        default value is filename of the PUF adjustment ratios.
        NOTE: when using custom ratios, set this argument to a DataFrame.
        NOTE: if specifying a DataFrame, set adjust_ratios to my_df defined as:
              my_df = pd.read_csv('<my_ratios.csv>', index_col=0).transpose()

    exact_calculations: boolean
        specifies whether or not exact tax calculations are done without
        any smoothing of stair-step provisions in income tax law;
        default value is false.

    Raises
    ------
    ValueError:
        if data is not the appropriate type.
        if taxpayer and spouse variables do not add up to filing-unit total.
        if dividends is less than qualified dividends.
        if gfactors is not None or a GrowFactors class instance.
        if start_year is not an integer.
        if files cannot be found.

    Returns
    -------
    class instance: Records

    Notes
    -----
    Typical usage when using PUF input data is as follows::

        recs = Records()

    which uses all the default parameters of the constructor, and
    therefore, imputed variables are generated to augment the data and
    initial-year grow factors are applied to the data.  There are
    situations in which you need to specify the values of the Record
    constructor's arguments, but be sure you know exactly what you are
    doing when attempting this.

    Use Records.cps_constructor() to get a Records object instantiated
    with CPS input data.
    """
    # suppress pylint warning about constructor having too many arguments:
    # pylint: disable=too-many-arguments
    # suppress pylint warnings about uppercase variable names:
    # pylint: disable=invalid-name
    # suppress pylint warnings about too many class instance attributes:
    # pylint: disable=too-many-instance-attributes

    PUFCSV_YEAR = 2011
    CPSCSV_YEAR = 2014

    PUF_WEIGHTS_FILENAME = 'puf_weights.csv.gz'
    PUF_RATIOS_FILENAME = 'puf_ratios.csv'
    CPS_WEIGHTS_FILENAME = 'cps_weights.csv.gz'
    CPS_RATIOS_FILENAME = None
    CODE_PATH = os.path.abspath(os.path.dirname(__file__))
    VARINFO_FILE_NAME = 'records_variables.json'
    VARINFO_FILE_PATH = CODE_PATH

    def __init__(self,
                 data='puf.csv',
                 start_year=PUFCSV_YEAR,
                 gfactors=GrowFactors(),
                 weights=PUF_WEIGHTS_FILENAME,
                 adjust_ratios=PUF_RATIOS_FILENAME,
                 exact_calculations=False):
        # pylint: disable=no-member,too-many-branches
        if isinstance(weights, str):
            weights = os.path.join(Records.CODE_PATH, weights)
        super().__init__(data, start_year, gfactors, weights)
        if data is None:
            return  # because there are no data
        # read adjustment ratios
        self.ADJ = None
        self._read_ratios(adjust_ratios)
        # specify exact value based on exact_calculations
        self.exact[:] = np.where(exact_calculations is True, 1, 0)
        # specify FLPDYR value based on start_year
        self.FLPDYR.fill(start_year)
        # check for valid MARS values
        if not np.all(np.logical_and(np.greater_equal(self.MARS, 1),
                                     np.less_equal(self.MARS, 5))):
            raise ValueError('not all MARS values in [1,5] range')
        # create variables derived from MARS, which is in MUST_READ_VARS
        self.num[:] = np.where(self.MARS == 2, 2, 1)
        self.sep[:] = np.where(self.MARS == 3, 2, 1)
        # check for valid EIC values
        if not np.all(np.logical_and(np.greater_equal(self.EIC, 0),
                                     np.less_equal(self.EIC, 3))):
            raise ValueError('not all EIC values in [0,3] range')
        # check that three sets of split-earnings variables have valid values
        msg = 'expression "{0} == {0}p + {0}s" is not true for every record'
        tol = 0.020001  # handles "%.2f" rounding errors
        if not np.allclose(self.e00200, (self.e00200p + self.e00200s),
                           rtol=0.0, atol=tol):
            raise ValueError(msg.format('e00200'))
        if not np.allclose(self.e00900, (self.e00900p + self.e00900s),
                           rtol=0.0, atol=tol):
            raise ValueError(msg.format('e00900'))
        if not np.allclose(self.e02100, (self.e02100p + self.e02100s),
                           rtol=0.0, atol=tol):
            raise ValueError(msg.format('e02100'))
        # check that spouse income variables have valid values
        nospouse = self.MARS != 2
        zeros = np.zeros_like(self.MARS[nospouse])
        msg = '{} is not always zero for non-married filing unit'
        if not np.allclose(self.e00200s[nospouse], zeros):
            raise ValueError(msg.format('e00200s'))
        if not np.allclose(self.e00900s[nospouse], zeros):
            raise ValueError(msg.format('e00900s'))
        if not np.allclose(self.e02100s[nospouse], zeros):
            raise ValueError(msg.format('e02100s'))
        if not np.allclose(self.k1bx14s[nospouse], zeros):
            raise ValueError(msg.format('k1bx14s'))
        # check that ordinary dividends are no less than qualified dividends
        other_dividends = np.maximum(0., self.e00600 - self.e00650)
        if not np.allclose(self.e00600, self.e00650 + other_dividends,
                           rtol=0.0, atol=tol):
            msg = 'expression "e00600 >= e00650" is not true for every record'
            raise ValueError(msg)
        del other_dividends
        # check that total pension income is no less than taxable pension inc
        nontaxable_pensions = np.maximum(0., self.e01500 - self.e01700)
        if not np.allclose(self.e01500, self.e01700 + nontaxable_pensions,
                           rtol=0.0, atol=tol):
            msg = 'expression "e01500 >= e01700" is not true for every record'
            raise ValueError(msg)
        del nontaxable_pensions
        # check that PT_SSTB_income has valid value
        if not np.all(np.logical_and(np.greater_equal(self.PT_SSTB_income, 0),
                                     np.less_equal(self.PT_SSTB_income, 1))):
            raise ValueError('not all PT_SSTB_income values are 0 or 1')

    @staticmethod
    def cps_constructor(data=None,
                        gfactors=GrowFactors(),
                        exact_calculations=False):
        """
        Static method returns a Records object instantiated with CPS
        input data.  This works in a analogous way to Records(), which
        returns a Records object instantiated with PUF input data.
        This is a convenience method that eliminates the need to
        specify all the details of the CPS input data just as the
        default values of the arguments of the Records class constructor
        eliminate the need to specify all the details of the PUF input
        data.
        """
        if data is None:
            data = os.path.join(Records.CODE_PATH, 'cps.csv.gz')
        if gfactors is None:
            weights = None
        else:
            weights = os.path.join(Records.CODE_PATH,
                                   Records.CPS_WEIGHTS_FILENAME)
        return Records(data=data,
                       start_year=Records.CPSCSV_YEAR,
                       gfactors=gfactors,
                       weights=weights,
                       adjust_ratios=Records.CPS_RATIOS_FILENAME,
                       exact_calculations=exact_calculations)

    def increment_year(self):
        """
        Add one to current year, and also does
        extrapolation, reweighting, adjusting for new current year.
        """
        super().increment_year()
        self.FLPDYR.fill(self.current_year)  # pylint: disable=no-member
        # apply variable adjustment ratios
        self._adjust(self.current_year)

    @staticmethod
    def read_cps_data():
        """
        Return data in cps.csv.gz as a Pandas DataFrame.
        """
        fname = os.path.join(Records.CODE_PATH, 'cps.csv.gz')
        if os.path.isfile(fname):
            cpsdf = pd.read_csv(fname)
        else:  # find file in conda package
            cpsdf = read_egg_csv(fname)  # pragma: no cover
        return cpsdf

    # ----- begin private methods of Records class -----

    def _extrapolate(self, year):
        """
        Apply to variables the grow factor values for specified calendar year.
        """
        # pylint: disable=too-many-statements,no-member
        # put values in local dictionary
        gfv = dict()
        for name in GrowFactors.VALID_NAMES:
            gfv[name] = self.gfactors.factor_value(name, year)
        # apply values to Records variables
        self.e00200 *= gfv['AWAGE']
        self.e00200p *= gfv['AWAGE']
        self.e00200s *= gfv['AWAGE']
        self.pencon_p *= gfv['AWAGE']
        self.pencon_s *= gfv['AWAGE']
        self.e00300 *= gfv['AINTS']
        self.e00400 *= gfv['AINTS']
        self.e00600 *= gfv['ADIVS']
        self.e00650 *= gfv['ADIVS']
        self.e00700 *= gfv['ATXPY']
        self.e00800 *= gfv['ATXPY']
        self.e00900s[:] = np.where(self.e00900s >= 0,
                                   self.e00900s * gfv['ASCHCI'],
                                   self.e00900s * gfv['ASCHCL'])
        self.e00900p[:] = np.where(self.e00900p >= 0,
                                   self.e00900p * gfv['ASCHCI'],
                                   self.e00900p * gfv['ASCHCL'])
        self.e00900[:] = self.e00900p + self.e00900s
        self.e01100 *= gfv['ACGNS']
        self.e01200 *= gfv['ACGNS']
        self.e01400 *= gfv['ATXPY']
        self.e01500 *= gfv['ATXPY']
        self.e01700 *= gfv['ATXPY']
        self.e02000[:] = np.where(self.e02000 >= 0,
                                  self.e02000 * gfv['ASCHEI'],
                                  self.e02000 * gfv['ASCHEL'])
        self.e02100 *= gfv['ASCHF']
        self.e02100p *= gfv['ASCHF']
        self.e02100s *= gfv['ASCHF']
        self.e02300 *= gfv['AUCOMP']
        self.e02400 *= gfv['ASOCSEC']
        self.e03150 *= gfv['ATXPY']
        self.e03210 *= gfv['ATXPY']
        self.e03220 *= gfv['ATXPY']
        self.e03230 *= gfv['ATXPY']
        self.e03270 *= gfv['ACPIM']
        self.e03240 *= gfv['ATXPY']
        self.e03290 *= gfv['ACPIM']
        self.e03300 *= gfv['ATXPY']
        self.e03400 *= gfv['ATXPY']
        self.e03500 *= gfv['ATXPY']
        self.e07240 *= gfv['ATXPY']
        self.e07260 *= gfv['ATXPY']
        self.e07300 *= gfv['ABOOK']
        self.e07400 *= gfv['ABOOK']
        self.p08000 *= gfv['ATXPY']
        self.e09700 *= gfv['ATXPY']
        self.e09800 *= gfv['ATXPY']
        self.e09900 *= gfv['ATXPY']
        self.e11200 *= gfv['ATXPY']
        # ITEMIZED DEDUCTIONS
        self.e17500 *= gfv['ACPIM']
        self.e18400 *= gfv['ATXPY']
        self.e18500 *= gfv['ATXPY']
        self.e19200 *= gfv['AIPD']
        self.e19800 *= gfv['ATXPY']
        self.e20100 *= gfv['ATXPY']
        self.e20400 *= gfv['ATXPY']
        self.g20500 *= gfv['ATXPY']
        # CAPITAL GAINS
        self.p22250 *= gfv['ACGNS']
        self.p23250 *= gfv['ACGNS']
        self.e24515 *= gfv['ACGNS']
        self.e24518 *= gfv['ACGNS']
        # SCHEDULE E
        self.e26270 *= gfv['ASCHEI']
        self.e27200 *= gfv['ASCHEI']
        self.k1bx14p *= gfv['ASCHEI']
        self.k1bx14s *= gfv['ASCHEI']
        # MISCELLANOUS SCHEDULES
        self.e07600 *= gfv['ATXPY']
        self.e32800 *= gfv['ATXPY']
        self.e58990 *= gfv['ATXPY']
        self.e62900 *= gfv['ATXPY']
        self.e87530 *= gfv['ATXPY']
        self.e87521 *= gfv['ATXPY']
        self.cmbtp *= gfv['ATXPY']
        # BENEFITS
        self.other_ben *= gfv['ABENOTHER']
        self.mcare_ben *= gfv['ABENMCARE']
        self.mcaid_ben *= gfv['ABENMCAID']
        self.ssi_ben *= gfv['ABENSSI']
        self.snap_ben *= gfv['ABENSNAP']
        self.wic_ben *= gfv['ABENWIC']
        self.housing_ben *= gfv['ABENHOUSING']
        self.tanf_ben *= gfv['ABENTANF']
        self.vet_ben *= gfv['ABENVET']

        # CAPITAL GAINS AT DEATH
        if year >= 2021:
            # TAXPAYERS' SHARE OF CAPITAL GAINS
            self.ltgains_wt = ((self.p23250 * self.s006) /
                                   np.sum(self.p23250 * self.s006))
            self.ltgains_wt[self.p23250 > 0] = (
                (self.p23250[self.p23250 > 0] * self.s006[self.p23250 > 0])
                / np.sum(self.p23250[self.p23250 > 0] *
                         self.s006[self.p23250 > 0]))
            self.ltgains_wt[self.p23250 <= 0] = 0
            # ASSIGN TOTAL REALIZATION AT DEATH TO TAXPAYERS
            # NOTE: VALUES ARE JCT CAP GAINS TAX EXPENDITURES FOR EXCLUSION AT DEATH/EMTR
            realization_at_death = {
                2021: 204253570343, 2022: 215627184994,
                2023: 226688939254, 2024: 235220544654,
                2025: 243736672365, 2026: 245533163473,
                2027: 254865207631, 2028: 264380671657,
                2029: 274178230759, 2030: 284077457991}
            self.gains_at_death = ((
                self.ltgains_wt * realization_at_death[year]) /
                                   (self.s006))

        # REFUNDABLE FIRST TIME HOMEBUYERS' CREDIT
        if year >= 2021:
            # TAXPAYERS' SHARE OF FIRST TIME HOMEBUYERS CREDIT
            self.fthb_wt = ((self.e00200 * self.s006) /
                                   np.sum(self.e00200 * self.s006))
            # ASSIGN FIRST TIME HOMEBUYERS' CREDIT TO TAXPAYERS
            # NOTE: VALUES ARE CALCULATED FROM WEIGHTED CREDIT VALUES (e11580) FROM 2009 CPS MATCHED PUF
            total_fthb_credit = {
                2021: 36235503285, 2022: 36477386218,
                2023: 36757344598, 2024: 37063748107,
                2025: 37366413299, 2026: 37631279957,
                2027: 37873993627, 2028: 38094000484,
                2029: 38307915270, 2030: 38515184160}
            self.fthb_credit_amt = ((
                self.fthb_wt * total_fthb_credit[year]) /
                            (self.s006))
                
        # NONREFUNDABLE INFORMAL CAREGIVER CREDIT
        if year >= 2021:
            # TAXPAYERS' SHARE OF DEPENDENT CARE EXPENSES
            self.icg_adj_wt = ((self.e32800 * self.s006) /
                                   np.sum(self.e32800 * self.s006))
            self.icg_eld_wt = ((self.e32800 * self.s006) /
                                   np.sum(self.e32800 * self.s006))
            self.icg_eld_wt[self.elderly_dependents>0] = (
                (self.e32800[self.elderly_dependents>0] * self.s006[self.elderly_dependents>0])
                / np.sum(self.e32800[self.elderly_dependents>0] *
                         self.s006[self.elderly_dependents>0]))
            self.icg_eld_wt[self.elderly_dependents<=0] = 0
            # ASSIGN DEPENDENT CARE EXPENSES TO TAXPAYERS
            # NOTE: VALUES ARE CALCULATED FROM PROJECTION OF EXP FOR LT CARE SERVICES FOR THE ELDERLY, CBO 1999.
            icg_elderly_expense = {
                2021: 58471879287, 2022: 61174211248,
                2023: 63672153635, 2024: 66429903978,
                2025: 69373388203, 2026: 72469958848,
                2027: 75672976680, 2028: 78866392318,
                2029: 81957475995, 2030: 85109190672}
            icg_adjusted_expense = {
                2021: 14617969822, 2022: 15293552812,
                2023: 15918038409, 2024: 16607475995,
                2025: 17343347051, 2026: 18117489712,
                2027: 18918244170, 2028: 19716598080,
                2029: 20489368999, 2030: 21277297668}
            self.icg_expense = (((self.icg_adj_wt * icg_elderly_expense[year]) +
                        (self.icg_adj_wt * icg_adjusted_expense[year]))/
                            (self.s006))
                            
        # NONREFUNDABLE FULL-ELECTRIC VEHICLE CREDIT
        if year >= 2021:
            # TAXPAYERS' SHARE OF FULL-ELECTRIC VEHICLE CREDIT CAPPED
            self.ev_wt = ((self.e00200 * self.s006) /
                                   np.sum(self.e00200 * self.s006))

            self.ev_wt[self.e00200 < 400000] = (
                 (self.e00200[self.e00200 < 400000] * self.s006[self.e00200 < 400000])
                 / np.sum(self.e00200[self.e00200 < 400000] *
                          self.s006[self.e00200 < 400000]))
            self.ev_wt[self.e00200 >= 400000] = 0
            # ASSIGN FULL-ELECTRIC VEHICLE CREDIT TO TAXPAYERS
            # NOTE: VALUES ARE PROJECTED OFF-MODEL SEE /SOURCES FOR MORE
            total_ev_credit = {
                2021: 1927126854, 2022: 1971463216,
                2023: 2064758865, 2024: 2267898548,
                2025: 2490472020, 2026: 2564513631,
                2027: 2656101550, 2028: 2784290330,
                2029: 2951747632, 2030: 3179428781}
            self.ev_credit_amt = ((
                self.ev_wt * total_ev_credit[year])/
                            (self.s006))
           
        # STUDENT LOAN DEBT FORGIVENESS
        if year >= 2021:
            # TAXPAYERS' SHARE OF FORGIVEN STUDENT LOANS
            self.studloan_wt = ((self.e03210 * self.s006) /
                                             np.sum(self.e03210 * self.s006))
            self.studloan_wt[self.e03210 > 0] = (
                      (self.e03210[self.e03210 > 0] * self.s006[self.e03210 > 0])
                      / np.sum(self.e03210[self.e03210 > 0] *
                                self.s006[self.e03210 > 0]))
            # ASSIGN FORGIVEN STUDENT LOANS TO TAXPAYERS
            # NOTE: VALUES ARE CALCULATED FROM PUBLIC SERVICE LOAN FORGIVENESS DATA, FSAID US DOE.
            total_studloan_debt = {
                2021: 4975784446, 2022: 5205744925,
                2023: 5418312454, 2024: 5652988873,
                2025: 5903470698, 2026: 6166979725,
                2027: 6439547095, 2028: 6711297345,
                2029: 6974339448, 2030: 7242541070}
            self.studloan_debt = ((
                self.studloan_wt * total_studloan_debt[year]) /
                            (self.s006))

        # AUTOMATIC ENROLLMENT IN IRAs CREDIT
        if year >= 2021:
            # TAXPAYERS' SHARE OF INVESTMENT INCOME
            self.autoira_wt = ((self.e58990 * self.s006) /
                                             np.sum(self.e58990 * self.s006))
            self.autoira_wt[self.e58990 > 0] = (
                          (self.e58990[self.e58990 > 0] * self.s006[self.e58990 > 0])
                          / np.sum(self.e58990[self.e58990 > 0] *
                                    self.s006[self.e58990 > 0]))
            # ASSIGN CREDIT TO TAXPAYERS
            # NOTE: VALUES ARE CALCULATED FROM JCT ESTIMATES, SEE /SOURCES FOR MORE.
            total_autoira_credit = {
                2021: 1536000000, 2022: 1606987660,
                2023: 1672606203, 2024: 1745049650,
                2025: 1822372148, 2026: 1903716079,
                2027: 1987856276, 2028: 2071744231,
                2029: 2152944025, 2030: 2235736536}
            self.ira_credit = ((
                self.autoira_wt * total_autoira_credit[year]) /
                                   (self.s006))
            
        # IMPUTE BUSINESS TAX BURDEN TO TAXPAYERS
        if year >= 2021:
            # TAXPAYERS' SHARE OF LABOR (WAGES)
            self.wage_wt = ((self.e00200 * self.s006) /
                                   np.sum(self.e00200 * self.s006))
            self.wage_wt[self.e00200 > 0] = (
                (self.e00200[self.e00200 > 0] * self.s006[self.e00200 > 0])
                / np.sum(self.e00200[self.e00200 > 0] *
                         self.s006[self.e00200 > 0]))
            # DEFINE CAPITAL
            self.capital = self.p22250 + self.p23250 + self.e00650 + self.e00650 + self.e00300 + self.e02000
            # TAXPAYERS' SHARE OF CAPITAL
            self.capital_wt = ((self.capital * self.s006) /
                                   np.sum(self.capital * self.s006))
            self.capital_wt[self.capital > 0] = (
                (self.capital[self.capital > 0] * self.s006[self.capital > 0])
                / np.sum(self.capital[self.capital > 0] *
                         self.s006[self.capital > 0]))
            # DEFINE BUSINESS TAX REVENUE ESTIMATES
            # NOTE: VALUES ARE ESTIMATED OFF-MODEL (SEE /SOURCES)
            business_revenue = {
                2021: 175598795234, 2022: 152920952040,
                2023: 171354346844, 2024: 183109348468,
                2025: 195411078716, 2026: 198933433969,
                2027: 211610303777, 2028: 220027985881,
                2029: 228211335814, 2030: 236808677430}
            # DEFINE CORPORATE INCOME TAX REVENUE PROJECTIONS
            # NOTE: VALUES ARE BASELINE CORPORATE REVENUE ESTIMATES FROM CBO PROJ. (UPDATED 09/20)
            corp_tax_revenue = {
                2021: 122754000000, 2022: 234076000000,
                2023: 289276000000, 2024: 318899000000,
                2025: 347329000000, 2026: 352277000000,
                2027: 355589000000, 2028: 368086000000,
                2029: 377565000000, 2030: 386582000000}
            # ASSIGN BUSINESS TAX REVENUE ESTIMATES AND CORPORATE TAX LIABILITY TO TAXPAYERS
            # WE ASSIGN 20% OF THE CORPORATE INCOME TAX BURDEN TO LABOR (WAGES)
            wage_rev_share = 0.2 * business_revenue[year]
            corp_taxliab_wage_share = 0.2 * corp_tax_revenue[year]
            # WE ASSIGN 80% OF THE CORPORATE INCOME TAX BURDEN TO CAPITAL
            cap_rev_share = 0.8 * business_revenue[year]
            corp_taxliab_cap_share = 0.8 * corp_tax_revenue[year]
            # IMPUTE TO TAXPAYERS
            self.busburden_w = ((
                self.wage_wt * wage_rev_share) /
                                   (self.s006))
            self.busburden_c = ((
                self.capital_wt * cap_rev_share) /
                                   (self.s006))
            self.corp_taxliab_w = ((
                self.wage_wt * corp_taxliab_wage_share) /
                                   (self.s006))
            self.corp_taxliab_c = ((
                self.wage_wt * corp_taxliab_cap_share) /
                                   (self.s006))
            self.business_burden = self.busburden_w + self.busburden_c
            self.corp_taxliab = self.corp_taxliab_w + self.corp_taxliab_c
            
        # IMPUTE ESTATE AND GIFT TAX BURDEN TO TAXPAYERS
        if year >= 2021:
            # TAXPAYERS' SHARE OF WAGES > 500000
            self.estate_wt = ((self.e00200 * self.s006) /
                                   np.sum(self.e00200 * self.s006))
            self.estate_wt[self.e00200 > 500000] = (
                (self.e00200[self.e00200 > 500000] * self.s006[self.e00200 > 500000])
                / np.sum(self.e00200[self.e00200 > 500000] *
                         self.s006[self.e00200 > 500000]))
            self.estate_wt[self.e00200 <= 500000] = 0
            # DEFINE ESTATE AND GIFT TAX REVENUE ESTIMATES
            # NOTE: VALUES ARE ESTIMATED OFF-MODEL (SEE /SOURCES)
            estate_gift_revenue = {
                2021: 26175000000, 2022: 27650000000,
                2023: 29091500000, 2024: 30599500000,
                2025: 31819646325, 2026: 30809910031,
                2027: 25247444835, 2028: 23859507263,
                2029: 24998461561, 2030: 26191784831}
            # IMPUTE TO TAXPAYERS
            self.estate_burden = ((
                self.estate_wt * estate_gift_revenue[year]) /
                                   (self.s006))
    
        # remove local dictionary
        del gfv

    def _adjust(self, year):
        """
        Adjust value of income variables to match SOI distributions
        Note: adjustment must leave variables as numpy.ndarray type
        """
        # pylint: disable=no-member
        if self.ADJ.size > 0:
            # Interest income
            self.e00300 *= self.ADJ['INT{}'.format(year)][self.agi_bin].values

    def _read_ratios(self, ratios):
        """
        Read Records adjustment ratios from file or
        use specified transposed/no-index DataFrame as ratios or
        create empty DataFrame if None
        """
        if ratios is None:
            setattr(self, 'ADJ', pd.DataFrame({'nothing': []}))
            return
        if isinstance(ratios, pd.DataFrame):
            assert 'INT2013' in ratios.columns  # check for transposed
            assert ratios.index.name is None  # check for no-index
            ADJ = ratios
        elif isinstance(ratios, str):
            ratios_path = os.path.join(Records.CODE_PATH, ratios)
            if os.path.isfile(ratios_path):
                ADJ = pd.read_csv(ratios_path,
                                  index_col=0)
            else:  # find file in conda package
                ADJ = read_egg_csv(os.path.basename(ratios_path),
                                   index_col=0)  # pragma: no cover
            ADJ = ADJ.transpose()
        else:
            msg = 'ratios is neither None nor a Pandas DataFrame nor a string'
            raise ValueError(msg)
        assert isinstance(ADJ, pd.DataFrame)
        if ADJ.index.name != 'agi_bin':
            ADJ.index.name = 'agi_bin'
        self.ADJ = pd.DataFrame()
        setattr(self, 'ADJ', ADJ.astype(np.float32))
        del ADJ
