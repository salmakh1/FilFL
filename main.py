import os

from omegaconf import DictConfig, OmegaConf
import hydra
# from train_clients import trainClients
from hydra.utils import instantiate

import logging
from hydra.core.hydra_config import HydraConfig


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="defaultConf")
def run(cfg: DictConfig) -> None:
    log.info("MAIN FUNCTION")
    log.info(OmegaConf.to_yaml(cfg))
    hydra_cfg = HydraConfig.get()
    log.info(OmegaConf.to_container(hydra_cfg.runtime.choices))
    hydr_cfg=OmegaConf.to_container(hydra_cfg.runtime.choices)
    print(cfg)
    clients=instantiate(cfg.train_schema,cfg=cfg, hydra_cfg=hydr_cfg, _recursive_= False)
    clients.train_clients()



if __name__ == '__main__':
    run()


