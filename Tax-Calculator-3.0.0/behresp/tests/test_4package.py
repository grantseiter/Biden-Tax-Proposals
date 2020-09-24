"""
Tests for package existence and dependencies consistency.
"""
# CODING-STYLE CHECKS:
# pycodestyle test_4package.py
# pylint --disable=locally-disabled test_4package.py

import os
import re
import subprocess
import yaml
import pytest


@pytest.mark.local
def test_for_package_existence():
    """
    Ensure that no conda behresp package is installed when running pytest.
    Primarily to help developers catch mistaken installations of behresp;
    the local mark prevents test from running on GitHub.
    """
    out = subprocess.check_output(['conda', 'list', 'behresp']).decode('ascii')
    envless_out = out.replace('behresp-dev', 'environment')
    if re.search('behresp', envless_out) is not None:
        assert 'behresp package' == 'installed'


def test_for_consistency(tests_path):
    """
    Ensure that there is consistency between environment.yml dependencies
    and conda.recipe/meta.yaml requirements.
    """
    # read conda.recipe/meta.yaml requirements
    meta_file = os.path.join(tests_path, '..', '..',
                             'conda.recipe', 'meta.yaml')
    with open(meta_file, 'r') as stream:
        meta = yaml.safe_load(stream)
    bld = set(meta['requirements']['build'])
    run = set(meta['requirements']['run'])
    # confirm conda.recipe/meta.yaml build and run requirements are the same
    assert bld == run
    # read environment.yml dependencies
    envr_file = os.path.join(tests_path, '..', '..',
                             'environment.yml')
    with open(envr_file, 'r') as stream:
        envr = yaml.safe_load(stream)
    env = set(envr['dependencies'])
    # confirm that environment and run packages are the same
    assert env == run