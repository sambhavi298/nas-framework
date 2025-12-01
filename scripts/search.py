# scripts/search.py

from nas.trainers.darts_trainer import DartsTrainer

if __name__ == "__main__":
    trainer = DartsTrainer(
        work_dir="./search_runs/run1",
        batch_size=64,
        epochs=30,
        use_amp=False,      # mixed precision optional
        unrolled=False,     # production: first-order DARTS
        save_every=5,
    )

    trainer.setup_data(data_root="./data", cifar_download=True)
    genotype = trainer.search()

    print("\n=== Final Discovered Architecture ===")
    print(genotype)
