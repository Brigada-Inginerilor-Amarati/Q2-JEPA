import hydra
from omegaconf import OmegaConf

import q2jepa.config  # noqa: F401  registers schema


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
