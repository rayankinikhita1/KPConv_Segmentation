
from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.datasets.S3DIS_Elecs import S3DISDataset, S3DISSampler, S3DISCollate

class S3DISDataModule(LightningDataModule):
    """A DataModule for S3DIS dataset."""

    def __init__(
        self,
        common: DictConfig
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.config = common

        self.batch_size_per_device = getattr(self.config, 'batch_num', 1)
        self.data_train: Optional[S3DISDataset] = None
        self.data_val: Optional[S3DISDataset] = None
        self.data_test: Optional[S3DISDataset] = None

        

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.batch_size_per_device % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size_per_device}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size_per_device // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = S3DISDataset(self.config, set='training', 
                                         use_potentials=self.config.use_potentials, 
                                         load_data=self.config.load_data)
            self.data_val = S3DISDataset(self.config, set='validation',
                                       use_potentials=self.config.use_potentials, 
                                       load_data=self.config.load_data)
            self.data_test = S3DISDataset(self.config, set='test',
                                        use_potentials=self.config.use_potentials, 
                                        load_data=self.config.load_data)

    def train_dataloader(self) -> DataLoader[Any]:
        sampler = S3DISSampler(self.data_train)
        dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=1,  # Always 1 for point cloud data
            sampler=sampler,
            collate_fn=S3DISCollate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )
        # Optionally calibrate sampler here if needed
        # sampler.calib_max_in(self.config, dataloader, verbose=True)
        sampler.calibration(dataloader, verbose=True)
        return dataloader

    def val_dataloader(self) -> DataLoader[Any]:
        sampler = S3DISSampler(self.data_val)
        dataloader = DataLoader(
            dataset=self.data_val,
            batch_size=1,
            sampler=sampler,
            collate_fn=S3DISCollate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )
        # Optionally calibrate sampler here if needed
        # sampler.calib_max_in(self.config, dataloader, verbose=True)
        sampler.calibration(dataloader, verbose=True)
        return dataloader

    def test_dataloader(self) -> DataLoader[Any]:
        sampler = S3DISSampler(self.data_test)
        dataloader = DataLoader(
            dataset=self.data_test,
            batch_size=1,
            sampler=sampler,
            collate_fn=S3DISCollate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )
        # Optionally calibrate sampler here if needed
        # sampler.calib_max_in(self.config, dataloader, verbose=True)
        sampler.calibration(dataloader, verbose=True)
        return dataloader


    def state_dict(self) -> Dict[Any, Any]:
        return {}

