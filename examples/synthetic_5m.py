from gamm import generate_synthetic_5m_close, make_gamm_dataset, gamm_defaults


def main():
    defaults = gamm_defaults()
    prices = generate_synthetic_5m_close(n_days=30, seed=0)
    data = make_gamm_dataset(
        prices,
        lag=int(defaults["lag"]),
        lag_sd=int(defaults["lag_sd"]),
        timestep=int(defaults["timestep"]),
    )
    print(f"prices={prices.shape} X={data.X.shape} y={data.y.shape}")


if __name__ == "__main__":
    main()
