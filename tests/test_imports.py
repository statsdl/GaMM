def test_core_import_does_not_require_model_extras():
    import gamm

    assert gamm.__version__ == "0.1.0"
