from gamm.experiment import run_synthetic_benchmark


def main():
    results = run_synthetic_benchmark(n_days=30, seed=0)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
