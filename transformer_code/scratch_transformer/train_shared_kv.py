from train_minimal import Config, ROOT, build_parser, train


def main():
    base = Config(
        attention_mode="shared_kv",
        save_dir=str(ROOT / "checkpoints" / "shared_kv"),
    )
    parser = build_parser(base)
    cfg = Config(**vars(parser.parse_args()))
    cfg.attention_mode = "shared_kv"
    train(cfg)


if __name__ == "__main__":
    main()
