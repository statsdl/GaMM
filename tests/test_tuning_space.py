from gamm.tuning import _clean_params


def test_clean_params_casts_hyperopt_values():
    clean = _clean_params({"ff_dim": 12.0, "n_block": 3.0, "dropout": 0.2, "activation_function": "gelu"})

    assert clean == {"ff_dim": 12, "n_block": 3, "dropout": 0.2, "activation_function": "gelu"}


def test_tuning_space_documents_expanded_activation_choices():
    from gamm.tuning import gamm_tuning_space

    space = gamm_tuning_space()

    assert set(space) == {"activation_function", "ff_dim", "dropout", "n_block"}
