"""
Tax-Calculator abstract base parameters class.
"""
# CODING-STYLE CHECKS:
# pycodestyle parameters.py
# pylint --disable=locally-disabled parameters.py

import os
import re
import abc
from collections import OrderedDict
import requests
import numpy as np
from taxcalc.utils import read_egg_json, json_to_dict


class Parameters():
    """
    Inherit from this class for Policy, Consumption, GrowDiff, and
    other groups of parameters that need to have a set_year method.
    Override this __init__ method and DEFAULTS_FILE_NAME and
    DEFAULTS_FILE_PATH in the inheriting class.
    """
    # pylint: disable=too-many-instance-attributes

    __metaclass__ = abc.ABCMeta

    DEFAULTS_FILE_NAME = None
    DEFAULTS_FILE_PATH = None

    def __init__(self):
        # convert JSON in DEFAULTS_FILE_NAME into self._vals dictionary
        assert self.DEFAULTS_FILE_NAME is not None
        assert self.DEFAULTS_FILE_PATH is not None
        file_path = os.path.join(self.DEFAULTS_FILE_PATH,
                                 self.DEFAULTS_FILE_NAME)
        if os.path.isfile(file_path):
            with open(file_path) as pfile:
                json_text = pfile.read()
            vals = json_to_dict(json_text)
        else:  # find file in conda package
            vals = read_egg_json(self.DEFAULTS_FILE_NAME)  # pragma: no cover
        # add leading underscore character to each parameter name
        self._vals = OrderedDict()
        for pname in vals:
            self._vals['_' + pname] = vals[pname]
        del vals
        # declare several scalar variables
        self._current_year = 0
        self._start_year = 0
        self._end_year = 0
        self._num_years = 0
        self._last_known_year = 0
        # declare optional _inflation_rates and _wage_growth_rates
        self._inflation_rates = list()
        self._wage_growth_rates = list()
        self._wage_indexed = None
        # declare removed and redefined parameters
        self._removed = None
        self._redefined = None
        # declare parameter warning/error variables
        self.parameter_warnings = ''
        self.parameter_errors = ''

    def initialize(self, start_year, num_years, last_known_year=None,
                   removed=None, redefined=None, wage_indexed=None):
        """
        Called from subclass __init__ function.
        """
        # pylint: disable=too-many-arguments
        # check arguments
        assert start_year >= 0
        assert num_years >= 1
        end_year = start_year + num_years - 1
        assert last_known_year is None or isinstance(last_known_year, int)
        assert removed is None or isinstance(removed, dict)
        assert redefined is None or isinstance(redefined, dict)
        assert wage_indexed is None or isinstance(wage_indexed, list)
        # remember arguments
        self._current_year = start_year
        self._start_year = start_year
        self._num_years = num_years
        self._end_year = end_year
        if last_known_year is None:
            self._last_known_year = start_year
        else:
            assert last_known_year >= start_year
            assert last_known_year <= end_year
            self._last_known_year = last_known_year
        if removed is None:
            self._removed = dict()
        else:
            self._removed = removed
        if redefined is None:
            self._redefined = dict()
        else:
            self._redefined = redefined
        if wage_indexed is None:
            self._wage_indexed = list()
        else:
            self._wage_indexed = wage_indexed
        # set default parameter values
        self._apply_cpi_offset_to_inflation_rates()
        self._set_default_vals()

    def inflation_rates(self):
        """
        Override this method in subclass when appropriate.
        """
        return self._inflation_rates

    def wage_growth_rates(self):
        """
        Override this method in subclass when appropriate.
        """
        return self._wage_growth_rates

    @property
    def num_years(self):
        """
        Parameters class number of parameter years property.
        """
        return self._num_years

    @property
    def current_year(self):
        """
        Parameters class current calendar year property.
        """
        return self._current_year

    @property
    def start_year(self):
        """
        Parameters class first parameter year property.
        """
        return self._start_year

    @property
    def last_known_year(self):
        """
        Parameters class last known parameter year property.
        """
        return self._last_known_year

    @property
    def end_year(self):
        """
        Parameters class last parameter year property.
        """
        return self._end_year

    def set_year(self, year):
        """
        Set parameters to their values for the specified calendar year.

        Parameters
        ----------
        year: integer
            calendar year for which to set current_year and parameter values

        Raises
        ------
        ValueError:
            if year is not in [start_year, end_year] range.

        Returns
        -------
        nothing: void
        """
        if year < self.start_year or year > self.end_year:
            msg = 'year {} passed to set_year() must be in [{},{}] range.'
            raise ValueError(msg.format(year, self.start_year, self.end_year))
        self._current_year = year
        iyr = year - self._start_year
        for name in self._vals:
            arr = getattr(self, name)
            setattr(self, name[1:], arr[iyr])

    def metadata(self):
        """
        Returns ordered dictionary of all parameter information based on
        DEFAULTS_FILE_NAME contents with each parameter's 'start_year',
        'value_yrs', and 'value' key values updated so that they contain
        just the current_year information.
        """
        mdata = OrderedDict()
        for pname, pdata in self._vals.items():
            name = pname[1:]
            mdata[name] = pdata
            mdata[name]['start_year'] = '{}'.format(self.current_year)
            mdata[name]['value_yrs'] = ['{}'.format(self.current_year)]
            valraw = getattr(self, name)
            if isinstance(valraw, np.ndarray):
                val = valraw.tolist()
            else:
                val = valraw
            mdata[name]['value'] = val
        return mdata

    @staticmethod
    def years_in_revision(revision):
        """
        Return list of years in specified revision dictionary, which is
        assumed to have a param:year:value format.
        """
        assert isinstance(revision, dict)
        years = list()
        for _, paramdata in revision.items():
            assert isinstance(paramdata, dict)
            for year, _ in paramdata.items():
                assert isinstance(year, int)
                if year not in years:
                    years.append(year)
        return years

    # ----- begin private methods of Parameters class -----

    def _set_default_vals(self, known_years=999999):
        """
        Called by initialize method and from some subclass methods.
        """
        # pylint: disable=too-many-branches,too-many-nested-blocks
        assert isinstance(known_years, (int, dict))
        if isinstance(known_years, int):
            known_years_is_int = True
        elif isinstance(known_years, dict):
            known_years_is_int = False
        for name, data in self._vals.items():
            valtype = data['value_type']
            values = data['value']
            indexed = data.get('indexed', False)
            if indexed:
                if name in self._wage_indexed:
                    index_rates = self.wage_growth_rates()
                else:
                    index_rates = self.inflation_rates()
                if known_years_is_int:
                    values = values[:known_years]
                else:
                    values = values[:known_years[name]]
            else:
                index_rates = None
            setattr(self, name,
                    self._expand_array(values, valtype,
                                       inflate=indexed,
                                       inflation_rates=index_rates,
                                       num_years=self._num_years))
        self.set_year(self._start_year)

    def _update(self, revision_, print_warnings, raise_errors):
        """
        Update parameters using specified revision_ dictionary and
        leave current_year unchanged.

        Parameters
        ----------
        revision_: parameter-changes dictionary in param:year:value format
            Each param primary key must be a string;
            each year secondary key must be an integer; and
            each value item must be either
              a real/integer/boolean/string value for a scalar parameter
              or
              a list of real/integer/boolean/string values for a vector param.

        print_warnings: boolean
            if True, prints warnings when parameter_warnings exists;
            if False, does not print warnings when parameter_warnings exists
                    and leaves warning handling to caller of _update method.

        raise_errors: boolean
            if True, raises ValueError when parameter_errors exists;
            if False, does not raise ValueError when parameter_errors exists
                    and leaves error handling to caller of _update method.

        Raises
        ------
        ValueError:
            if revision_ is not a dictionary.
            if each revision_ primary key is not a valid parameter name.
            if each revision_ secondary key is not an integet.
            if minimum year in revision_ is less than current_year.
            if maximum year in revision_ is greater than end_year.
            if _validate_names_types generates errors
            if _validate_values generates errors and raise_errors is True

        Returns
        -------
        nothing: void
        """
        # pylint: disable=too-many-locals,too-many-branches
        # check revisions_ type and whether empty
        if not isinstance(revision_, dict):
            raise ValueError('ERROR: YYYY PARAM revision_ is not a dictionary')
        if not revision_:
            return  # no revisions provided to update parameters
        # convert revision_ to revision with year:param:value format
        revision = dict()
        for name, namedata in revision_.items():
            if not isinstance(name, str):
                msg = 'ERROR: KEY {} is not a string parameter name'
                raise ValueError(msg.format(name))
            if not isinstance(namedata, dict):
                msg = 'ERROR: KEY {} VAL {} is not a year:value dictionary'
                raise ValueError(msg.format(name, namedata))
            for year, yeardata in namedata.items():
                if not isinstance(year, int):
                    msg = 'ERROR: KEY {} YEAR {} is not an integer year'
                    raise ValueError(msg.format(name, year))
                if year not in revision:
                    revision[year] = dict()
                revision[year][name] = yeardata
        # check range of revision years
        revision_years = list(revision.keys())
        first_revision_year = min(revision_years)
        if first_revision_year < self.current_year:
            msg = 'ERROR: {} YEAR revision provision in YEAR < current_year={}'
            raise ValueError(msg.format(first_revision_year,
                                        self.current_year))
        last_revision_year = max(revision_years)
        if last_revision_year > self.end_year:
            msg = 'ERROR: {} YEAR revision provision in YEAR > end_year={}'
            raise ValueError(msg.format(last_revision_year, self.end_year))
        # add leading underscore character to each parameter name in revision
        revision = Parameters._add_underscores(revision)
        # add brackets around each value element in revision
        revision = Parameters._add_brackets(revision)
        # validate revision parameter names and types
        self.parameter_warnings = ''
        self.parameter_errors = ''
        self._validate_names_types(revision)
        if self.parameter_errors:
            raise ValueError(self.parameter_errors)
        # optionally apply CPI_offset to inflation_rates and re-initialize
        known_years = self._apply_cpi_offset_in_revision(revision)
        if known_years is not None:
            self._set_default_vals(known_years=known_years)
        # implement the revision year by year
        precall_current_year = self.current_year
        revision_parameters = set()
        for year in sorted(revision_years):
            self.set_year(year)
            revision_parameters.update(revision[year].keys())
            self._update_for_year({year: revision[year]})
        self.set_year(precall_current_year)
        # validate revision parameter values
        self._validate_values(revision_parameters)
        if self.parameter_warnings and print_warnings:
            print(self.parameter_warnings)
        if self.parameter_errors and raise_errors:
            raise ValueError('\n' + self.parameter_errors)

    def _update_for_year(self, year_mods):
        """
        Private method used by Parameters._update method.
        """
        # pylint: disable=too-many-locals
        # check YEAR value in the single YEAR:MODS dictionary parameter
        assert isinstance(year_mods, dict)
        assert len(year_mods.keys()) == 1
        year = list(year_mods.keys())[0]
        assert year == self.current_year
        # check that MODS is a dictionary
        assert isinstance(year_mods[year], dict)
        # implement reform provisions included in the single YEAR:MODS pair
        num_years_to_expand = (self.start_year + self.num_years) - year
        all_names = set(year_mods[year].keys())  # no duplicate keys in a dict
        used_names = set()  # set of used parameter names in MODS dict
        for name, values in year_mods[year].items():
            # determine indexing status of parameter with name for year
            if name.endswith('-indexed'):
                continue  # handle elsewhere in this method
            vals_indexed = self._vals[name].get('indexed', False)
            valtype = self._vals[name].get('value_type')
            name_plus_indexed = name + '-indexed'
            if name_plus_indexed in year_mods[year].keys():
                used_names.add(name_plus_indexed)
                indexed = year_mods[year].get(name_plus_indexed)
                self._vals[name]['indexed'] = indexed  # remember status
            else:
                indexed = vals_indexed
            # set post-reform values of parameter with name
            used_names.add(name)
            cval = getattr(self, name, None)
            wage_indexed_param = name in self._wage_indexed
            index_rates = self._indexing_rates_for_update(wage_indexed_param,
                                                          year,
                                                          num_years_to_expand)
            nval = self._expand_array(values, valtype,
                                      inflate=indexed,
                                      inflation_rates=index_rates,
                                      num_years=num_years_to_expand)
            cval[(year - self.start_year):] = nval
        # handle unused parameter names, all of which end in -indexed, but
        # some parameter names ending in -indexed were handled above
        unused_names = all_names - used_names
        for name in unused_names:
            used_names.add(name)
            pname = name[:-8]  # root parameter name
            pindexed = year_mods[year][name]
            self._vals[pname]['indexed'] = pindexed  # remember status
            cval = getattr(self, pname, None)
            pvalues = [cval[year - self.start_year]]
            wage_indexed_param = pname in self._wage_indexed
            index_rates = self._indexing_rates_for_update(wage_indexed_param,
                                                          year,
                                                          num_years_to_expand)
            valtype = self._vals[pname].get('value_type')
            nval = self._expand_array(pvalues, valtype,
                                      inflate=pindexed,
                                      inflation_rates=index_rates,
                                      num_years=num_years_to_expand)
            cval[(year - self.start_year):] = nval
        # confirm that all names have been used
        assert len(used_names) == len(all_names)
        # implement updated parameters for year
        self.set_year(year)

    def _validate_names_types(self, revision):
        """
        Check validity of parameter names and parameter types used
        in the specified revision dictionary, which is assumed to
        have a year:param:value format
        """
        # pylint: disable=too-many-branches,too-many-nested-blocks
        # pylint: disable=too-many-statements,too-many-locals
        assert isinstance(self._vals, dict)
        param_names = set(self._vals.keys())
        for year in sorted(revision.keys()):
            for name in revision[year]:
                if name.endswith('-indexed'):
                    if isinstance(revision[year][name], bool):
                        pname = name[:-8]  # root parameter name
                        if pname not in param_names:
                            if pname in self._removed:
                                msg = self._removed[pname]
                            else:
                                msg = 'is an unknown parameter name'
                            self.parameter_errors += (
                                'ERROR: {} {} '.format(year, name[1:]) +
                                msg + '\n'
                            )
                        else:
                            # check if root parameter is indexable
                            indexable = self._vals[pname].get('indexable',
                                                              False)
                            if not indexable:
                                msg = '{} {} parameter is not indexable'
                                self.parameter_errors += (
                                    'ERROR: ' +
                                    msg.format(year, pname[1:]) + '\n'
                                )
                    else:
                        msg = '{} {} parameter is not true or false'
                        self.parameter_errors += (
                            'ERROR: ' + msg.format(year, name[1:]) + '\n'
                        )
                else:  # if name does not end with '-indexed'
                    if name not in param_names:
                        if name in self._removed:
                            msg = self._removed[name]
                        else:
                            msg = 'is an unknown parameter name'
                        self.parameter_errors += (
                            'ERROR: {} {} '.format(year, name[1:]) + msg + '\n'
                        )
                    else:
                        # check parameter value type avoiding use of isinstance
                        # because isinstance(True, (int,float)) is True, which
                        # makes it impossible to check float parameters
                        valtype = self._vals[name]['value_type']
                        assert isinstance(revision[year][name], list)
                        pvalue = revision[year][name][0]
                        if isinstance(pvalue, list):
                            scalar = False  # parameter value is a list
                            if not self._vals[name].get('vi_vals', []):
                                msg = ('{} {} with value {} '
                                       'should be a scalar parameter')
                                self.parameter_errors += (
                                    'ERROR: ' +
                                    msg.format(year, name[1:], pvalue) +
                                    '\n'
                                )
                                # following is not true but is needed to
                                # avoid errors below
                                scalar = True
                        else:
                            scalar = True  # parameter value is a scalar
                            if self._vals[name].get('vi_vals', []):
                                msg = ('{} {} with value {} '
                                       'should be a vector parameter')
                                self.parameter_errors += (
                                    'ERROR: ' +
                                    msg.format(year, name[1:], pvalue) +
                                    '\n'
                                )
                            pvalue = [pvalue]  # make scalar a single-item list
                        # pylint: disable=consider-using-enumerate
                        for idx in range(0, len(pvalue)):
                            if scalar:
                                pname = name
                            else:
                                col = self._vals[name]['vi_vals'][idx]
                                pname = '{}[{}]'.format(name, col)
                            pval = pvalue[idx]
                            # pylint: disable=unidiomatic-typecheck
                            if valtype == 'real':
                                if type(pval) != float and type(pval) != int:
                                    msg = '{} {} value {} is not a number'
                                    self.parameter_errors += (
                                        'ERROR: ' +
                                        msg.format(year, pname[1:], pval) +
                                        '\n'
                                    )
                            elif valtype == 'boolean':
                                if type(pval) != bool:
                                    msg = '{} {} value {} is not boolean'
                                    self.parameter_errors += (
                                        'ERROR: ' +
                                        msg.format(year, pname[1:], pval) +
                                        '\n'
                                    )
                            elif valtype == 'integer':
                                if type(pval) != int:
                                    msg = '{} {} value {} is not integer'
                                    self.parameter_errors += (
                                        'ERROR: ' +
                                        msg.format(year, pname[1:], pval) +
                                        '\n'
                                    )
                            elif valtype == 'string':
                                if type(pval) != str:
                                    msg = '{} {} value {} is not a string'
                                    self.parameter_errors += (
                                        'ERROR: ' +
                                        msg.format(year, pname[1:], pval) +
                                        '\n'
                                    )
        del param_names

    def _validate_values(self, parameters_set):
        """
        Check values of parameters in specified parameter_set using
        range information from DEFAULTS_FILE_NAME JSON file.
        """
        # pylint: disable=too-many-statements,too-many-locals
        # pylint: disable=too-many-branches,too-many-nested-blocks
        assert isinstance(parameters_set, set)
        parameters = sorted(parameters_set)
        syr = self.start_year
        for pname in parameters:
            if pname.endswith('-indexed'):
                continue  # *-indexed parameter values validated elsewhere
            if pname in self._redefined:
                msg = self._redefined[pname]
                self.parameter_warnings += msg + '\n'
            pvalue = getattr(self, pname)
            if self._vals[pname]['value_type'] == 'string':
                valid_options = self._vals[pname]['valid_values']['options']
                for idx in np.ndindex(pvalue.shape):
                    if pvalue[idx] not in valid_options:
                        msg = "{} {} value '{}' not in {}"
                        fullmsg = '{}: {}\n'.format(
                            'ERROR',
                            msg.format(idx[0] + syr,
                                       pname[1:],
                                       pvalue[idx],
                                       valid_options)
                        )
                        self.parameter_errors += fullmsg
            else:  # parameter does not have string type
                for vop, vval in self._vals[pname]['valid_values'].items():
                    if isinstance(vval, str):
                        vvalue = getattr(self, '_' + vval)
                    else:
                        vvalue = np.full(pvalue.shape, vval)
                    assert pvalue.shape == vvalue.shape
                    assert len(pvalue.shape) <= 2
                    if len(pvalue.shape) == 2:
                        scalar = False  # parameter value is a vector
                    else:
                        scalar = True  # parameter value is a scalar
                    for idx in np.ndindex(pvalue.shape):
                        out_of_range = False
                        if vop == 'min' and pvalue[idx] < vvalue[idx]:
                            out_of_range = True
                            msg = '{} {} value {} < min value {}'
                            extra = self._vals[pname].get('invalid_minmsg', '')
                            if extra:
                                msg += ' {}'.format(extra)
                        if vop == 'max' and pvalue[idx] > vvalue[idx]:
                            out_of_range = True
                            msg = '{} {} value {} > max value {}'
                            extra = self._vals[pname].get('invalid_maxmsg', '')
                            if extra:
                                msg += ' {}'.format(extra)
                        if out_of_range:
                            action = self._vals[pname].get('invalid_action',
                                                           'stop')
                            if scalar:
                                name = pname
                            else:
                                col = self._vals[pname]['vi_vals'][idx[1]]
                                name = '{}[{}]'.format(pname, col)
                                if extra:
                                    msg += '[{}]'.format(col)
                            if action == 'warn':
                                fullmsg = '{}: {}\n'.format(
                                    'WARNING',
                                    msg.format(idx[0] + syr,
                                               name,
                                               pvalue[idx],
                                               vvalue[idx])
                                )
                                self.parameter_warnings += fullmsg
                            if action == 'stop':
                                fullmsg = '{}: {}\n'.format(
                                    'ERROR',
                                    msg.format(idx[0] + syr,
                                               name[1:],
                                               pvalue[idx],
                                               vvalue[idx])
                                )
                                self.parameter_errors += fullmsg
        del parameters

    STRING_DTYPE = 'U16'

    @staticmethod
    def _expand_array(xxx, xxx_type, inflate, inflation_rates, num_years):
        """
        Private method called only within this abstract base class.
        Dispatch to either _expand_1d or _expand_2d given dimension of xxx.

        Parameters
        ----------
        xxx : value to expand
              xxx must be either a scalar list or a 1D numpy array, or
              xxx must be either a list of scalar lists or a 2D numpy array

        xxx_type : string ('real', 'boolean', 'integer', 'string')

        inflate: boolean
            As we expand, inflate values if this is True, otherwise, just copy

        inflation_rates: list of inflation rates
            Annual decimal inflation rates

        num_years: int
            Number of budget years to expand

        Returns
        -------
        expanded numpy array with specified type
        """
        assert isinstance(xxx, (list, np.ndarray))
        if isinstance(xxx, list):
            if xxx_type == 'real':
                xxx = np.array(xxx, np.float64)
            elif xxx_type == 'boolean':
                xxx = np.array(xxx, np.bool_)
            elif xxx_type == 'integer':
                xxx = np.array(xxx, np.int16)
            elif xxx_type == 'string':
                xxx = np.array(xxx, np.dtype(Parameters.STRING_DTYPE))
                assert len(xxx.shape) == 1, \
                    'string parameters must be scalar (not vector)'
        dim = len(xxx.shape)
        assert dim in (1, 2)
        if dim == 1:
            return Parameters._expand_1d(xxx, inflate, inflation_rates,
                                         num_years)
        return Parameters._expand_2d(xxx, inflate, inflation_rates,
                                     num_years)

    @staticmethod
    def _expand_1d(xxx, inflate, inflation_rates, num_years):
        """
        Private method called only from _expand_array method.
        Expand the given data xxx to account for given number of budget years.
        If necessary, pad out additional years by increasing the last given
        year using the given inflation_rates list.
        """
        if not isinstance(xxx, np.ndarray):
            raise ValueError('_expand_1d expects xxx to be a numpy array')
        if len(xxx) >= num_years:
            return xxx
        string_type = xxx.dtype == Parameters.STRING_DTYPE
        if string_type:
            ans = np.array(['' for i in range(0, num_years)],
                           dtype=xxx.dtype)
        else:
            ans = np.zeros(num_years, dtype=xxx.dtype)
        ans[:len(xxx)] = xxx
        if string_type:
            extra = [str(xxx[-1]) for i in
                     range(1, num_years - len(xxx) + 1)]
        else:
            if inflate:
                extra = []
                cur = xxx[-1]
                for i in range(0, num_years - len(xxx)):
                    cur *= (1. + inflation_rates[i + len(xxx) - 1])
                    cur = round(cur, 2) if cur < 9e99 else 9e99
                    extra.append(cur)
            else:
                extra = [float(xxx[-1]) for i in
                         range(1, num_years - len(xxx) + 1)]
        ans[len(xxx):] = extra
        return ans

    @staticmethod
    def _expand_2d(xxx, inflate, inflation_rates, num_years):
        """
        Private method called only from _expand_array method.
        Expand the given data to account for the given number of budget years.
        For 2D arrays, we expand out the number of rows until we have num_years
        number of rows. For each expanded row, we inflate using the given
        inflation rates list.
        """
        if not isinstance(xxx, np.ndarray):
            raise ValueError('_expand_2d expects xxx to be a numpy array')
        if xxx.shape[0] >= num_years:
            return xxx
        ans = np.zeros((num_years, xxx.shape[1]), dtype=xxx.dtype)
        ans[:len(xxx), :] = xxx
        for i in range(xxx.shape[0], ans.shape[0]):
            for j in range(ans.shape[1]):
                if inflate:
                    cur = (ans[i - 1, j] *
                           (1. + inflation_rates[i - 1]))
                    cur = round(cur, 2) if cur < 9e99 else 9e99
                    ans[i, j] = cur
                else:
                    ans[i, j] = ans[i - 1, j]
        return ans

    def _indexing_rates_for_update(self, param_is_wage_indexed,
                                   calyear, num_years_to_expand):
        """
        Private method called only by the private Parameter._update method.
        """
        if param_is_wage_indexed:
            rates = self.wage_growth_rates()
        else:
            rates = self.inflation_rates()
        if rates:
            expanded_rates = [rates[(calyear - self.start_year) + i]
                              for i in range(0, num_years_to_expand)]
            return expanded_rates
        return None

    @staticmethod
    def _add_underscores(update_dict):
        """
        Returns dictionary that adds leading underscore character to
        each parameter name in specified update_dict, which is assumed
        to have a year:param:value format.
        """
        updict = dict()
        for year, yeardata in update_dict.items():
            updict[year] = dict()
            for pname, pvalue in yeardata.items():
                updict[year]['_' + pname] = pvalue
        return updict

    @staticmethod
    def _add_brackets(update_dict):
        """
        Returns dictionary that adds brackets around each
        data element (value) in specified update_dict, which
        is assumed to have a year:param:value format.
        """
        updict = dict()
        for year, yeardata in update_dict.items():
            updict[year] = dict()
            for pname, pvalue in yeardata.items():
                if pname.endswith('-indexed'):
                    updict[year][pname] = pvalue  # no added brackets
                else:
                    updict[year][pname] = [pvalue]
        return updict

    def _apply_cpi_offset_to_inflation_rates(self):
        """
        Called from Parameters.initialize method.
        Does nothing if CPI_offset parameter is not in self._vals dictionary.
        """
        if '_CPI_offset' not in self._vals:
            return
        nyrs = self.num_years
        ovalues = self._vals['_CPI_offset']['value']
        if len(ovalues) < nyrs:  # extrapolate last known value
            ovalues = ovalues + ovalues[-1:] * (nyrs - len(ovalues))
        for idx in range(0, nyrs):
            infrate = round(self._inflation_rates[idx] + ovalues[idx], 6)
            self._inflation_rates[idx] = infrate

    def _apply_cpi_offset_in_revision(self, revision):
        """
        Apply CPI offset to inflation rates and
        revert indexed parameter values in preparation for re-indexing.
        Also, return known_years which is dictionary with indexed policy
        parameter names as keys and known_years as values.  For indexed
        parameters included in revision, the known_years value is equal to:
        (first_cpi_offset_year - start_year + 1).  For indexed parameters
        not included in revision, the known_years value is equal to:
        (max(first_cpi_offset_year, last_known_year) - start_year + 1).
        """
        # pylint: disable=too-many-branches
        # determine if CPI_offset is in specified revision; if not, return
        cpi_offset_in_revision = False
        for year in revision:
            for name in revision[year]:
                if name == '_CPI_offset':
                    cpi_offset_in_revision = True
                    break  # out of loop
        if not cpi_offset_in_revision:
            return None
        # extrapolate CPI_offset revision
        self.set_year(self.start_year)
        first_cpi_offset_year = 0
        for year in sorted(revision.keys()):
            self.set_year(year)
            if '_CPI_offset' in revision[year]:
                if first_cpi_offset_year == 0:
                    first_cpi_offset_year = year
                orevision = {'_CPI_offset': revision[year]['_CPI_offset']}
                self._update_for_year({year: orevision})
        self.set_year(self.start_year)
        assert first_cpi_offset_year > 0
        # adjust inflation rates
        cpi_offset = getattr(self, '_CPI_offset')
        first_cpi_offset_ix = first_cpi_offset_year - self.start_year
        for idx in range(first_cpi_offset_ix, self.num_years):
            infrate = round(self._inflation_rates[idx] + cpi_offset[idx], 6)
            self._inflation_rates[idx] = infrate
        # revert indexed parameter values to policy_current_law.json values
        for name in self._vals.keys():
            if self._vals[name]['indexed']:
                setattr(self, name, self._vals[name]['value'])
        # construct and return known_years dictionary
        known_years = dict()
        kyrs_in_revision = (first_cpi_offset_year - self.start_year + 1)
        kyrs_not_in_revision = (
            max(first_cpi_offset_year, self.last_known_year) -
            self.start_year + 1
        )
        for year in sorted(revision.keys()):
            for name in revision[year]:
                if name.endswith('-indexed'):
                    name = name[:-8]
                if self._vals[name]['indexed']:
                    if name not in known_years:
                        known_years[name] = kyrs_in_revision
        for name in self._vals.keys():
            if self._vals[name]['indexed']:
                if name not in known_years:
                    known_years[name] = kyrs_not_in_revision
        return known_years

    @staticmethod
    def _read_json_revision(obj, topkey):
        """
        Read JSON revision specified by obj and topkey
        returning a single revision dictionary suitable for
        use with the Parameters._update method.

        The obj function argument can be None or a string, where the
        string contains a local filename, a URL beginning with 'http'
        pointing to a valid JSON file hosted online, or valid JSON
        text.

        The topkey argument must be a string containing the top-level
        key in a compound-revision JSON text for which a revision
        dictionary is returned.  If the specified topkey is not among
        the top-level JSON keys, the obj is assumed to be a
        non-compound-revision JSON text for the specified topkey.
        """
        # embedded function used only in _read_json_revision staticmethod
        def convert_year_to_int(syr_dict):
            """
            Converts specified syr_dict, which has string years as secondary
            keys, into a dictionary with the same structure but having integer
            years as secondary keys.
            """
            iyr_dict = dict()
            for pkey, sdict in syr_dict.items():
                assert isinstance(pkey, str)
                iyr_dict[pkey] = dict()
                assert isinstance(sdict, dict)
                for skey, val in sdict.items():
                    assert isinstance(skey, str)
                    year = int(skey)
                    iyr_dict[pkey][year] = val
            return iyr_dict
        # end of embedded function
        # process the main function arguments
        if obj is None:
            return dict()
        if not isinstance(obj, str):
            raise ValueError('obj is neither None nor a string')
        if not isinstance(topkey, str):
            raise ValueError('topkey={} is not a string'.format(topkey))
        if os.path.isfile(obj):
            if not obj.endswith('.json'):
                msg = 'obj does not end with ".json": {}'
                raise ValueError(msg.format(obj))
            txt = open(obj, 'r').read()
        elif obj.startswith('http'):
            if not obj.endswith('.json'):
                msg = 'obj does not end with ".json": {}'
                raise ValueError(msg.format(obj))
            req = requests.get(obj)
            req.raise_for_status()
            txt = req.text
        else:
            txt = obj
        # strip out //-comments without changing line numbers
        json_txt = re.sub('//.*', ' ', txt)
        # convert JSON text into a Python dictionary
        full_dict = json_to_dict(json_txt)
        # check top-level key contents of dictionary
        if topkey in full_dict.keys():
            single_dict = full_dict[topkey]
        else:
            single_dict = full_dict
        # convert string year to integer year in dictionary and return
        return convert_year_to_int(single_dict)
