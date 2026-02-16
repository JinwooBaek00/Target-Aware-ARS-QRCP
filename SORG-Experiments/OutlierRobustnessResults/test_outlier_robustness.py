import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

# Mock the SORG and Baselines modules to avoid ImportError
sys.modules['SORG'] = MagicMock()
sys.modules['Baselines'] = MagicMock()
from SORG import SORG
from Baselines import BL

# Mock the SORG class behavior
mock_sorg_instance = MagicMock()
mock_sorg_instance.get_support.return_value = list(range(50))  
SORG.return_value = mock_sorg_instance

from Outlier_Robustness import (
    generate_outliers,
    generate_outlier_labels,
    select_sorg_prefixes,
    train_eval,
    E5Config,
)

# Test generate_outliers
def test_generate_outliers():
    rng = np.random.RandomState(42)
    n_out = 10
    d = 5
    pool_median = 1.0
    multiplier = 2.0
    df = 2.0
    outliers = generate_outliers(rng, n_out, d, pool_median, multiplier, df)
    assert outliers.shape == (n_out, d)
    assert np.allclose(np.linalg.norm(outliers, axis=1).mean(), pool_median * multiplier, atol=1e-1)

# Test generate_outlier_labels
def test_generate_outlier_labels():
    rng = np.random.RandomState(42)
    n_out = 10
    n_classes = 3
    labels = generate_outlier_labels(rng, n_out, n_classes)
    assert labels.shape == (n_out,)
    assert np.all(labels < n_classes)

# Test select_sorg_prefixes
def test_select_sorg_prefixes():
    Xp = np.random.rand(100, 10)
    k_list = (10, 20, 50)
    p_norm = 2.0
    prefixes = select_sorg_prefixes(Xp, k_list, p_norm)
    assert isinstance(prefixes, dict)
    for k in k_list:
        if k <= Xp.shape[0]:
            assert k in prefixes
            assert len(prefixes[k]) == k

# Test train_eval
def test_train_eval():
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    acc, train_time = train_eval(X_train, y_train, X_test, y_test)
    assert 0.0 <= acc <= 1.0
    assert train_time > 0

# Test E5Config defaults
def test_e5config_defaults():
    cfg = E5Config()
    assert cfg.dataset == "cifar100"
    assert cfg.encoder == "resnet18"
    assert cfg.device in ["cuda", "cpu"]
    assert cfg.pool_size == 6000
    assert cfg.outlier_multiplier == 50.0
    assert cfg.outlier_df == 2.0