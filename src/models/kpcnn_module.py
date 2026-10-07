from typing import Optional
import torch
from lightning.pytorch import LightningModule
from .architectures import KPFCNN
from omegaconf import DictConfig
class KPCNNLitModule(LightningModule):
    
    """
    PyTorch Lightning wrapper for KPCNN architecture for point cloud segmentation.
    Supports multiple datasets including S3DIS and SemanticKITTI with configurable parameters.
    """
    # TODO: Simplify intialization by using a config object
    def __init__(
        self,
        common: DictConfig, label_values: Optional[list] = None, ignored_labels: Optional[list] = None, num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.config = common

        # Ensure num_classes matches the actual number of classes in the dataset
        if label_values is not None and num_classes is None:
            self.num_classes = len(label_values)
        from torchmetrics.classification import JaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
        self.train_miou = JaccardIndex(task="multiclass", num_classes=self.num_classes)
        self.val_miou = JaccardIndex(task="multiclass", num_classes=self.num_classes)
        self.test_miou = JaccardIndex(task="multiclass", num_classes=self.num_classes)
        self.train_precision = MulticlassPrecision(num_classes=self.num_classes, average=None)
        self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average=None)
        self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average=None)
        self.train_recall = MulticlassRecall(num_classes=self.num_classes, average=None)
        self.val_recall = MulticlassRecall(num_classes=self.num_classes, average=None)
        self.test_recall = MulticlassRecall(num_classes=self.num_classes, average=None)
        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
        self.train_confmat = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.train_class_iou = JaccardIndex(task="multiclass", num_classes=self.num_classes, average=None)
        self.val_class_iou = JaccardIndex(task="multiclass", num_classes=self.num_classes, average=None)
        self.test_class_iou = JaccardIndex(task="multiclass", num_classes=self.num_classes, average=None)


        self.model = KPFCNN(self.config, label_values, ignored_labels)
        self.lr = self.config.learning_rate  # Use learning_rate parameter
    

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        preds = outputs.argmax(dim=1)
        labels = batch.labels
        loss = self.model.loss(outputs, labels)
        acc = self.model.accuracy(outputs, labels)
        mask = (labels >= 0) & (labels < self.num_classes)
        preds_masked = preds[mask]
        labels_masked = labels[mask]
        miou = self.train_miou(preds_masked, labels_masked)
        precision = self.train_precision(preds_masked, labels_masked)
        recall = self.train_recall(preds_masked, labels_masked)
        f1 = self.train_f1(preds_masked, labels_masked)
        confmat = self.train_confmat(preds_masked, labels_masked)
        batch_size = batch.features.shape[0] if hasattr(batch, 'features') else None
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/miou", miou, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        # Helper to get class label string
        train_class_iou = self.train_class_iou(preds_masked, labels_masked)
        num_labels = len(self.hparams.label_values) if hasattr(self.hparams, 'label_values') and self.hparams.label_values is not None else self.num_classes
        for i in range(num_labels):
            if i < len(train_class_iou):
                self.log(f"train/class_iou_{i}", train_class_iou[i], on_step=False, on_epoch=True, batch_size=batch_size)

        def get_class_name(i):
            if hasattr(self.hparams, 'label_names') and self.hparams.label_names is not None:
                if i < len(self.hparams.label_names):
                    return str(self.hparams.label_names[i])
            if hasattr(self.hparams, 'label_values') and self.hparams.label_values is not None:
                if i < len(self.hparams.label_values):
                    return str(self.hparams.label_values[i])
            return str(i)
        
        for i in range(num_labels):
            class_name = get_class_name(i)
            if i < len(precision):
                self.log(f"train/precision_class_{i}_{class_name}", precision[i], on_step=False, on_epoch=True, batch_size=batch_size)
            if i < len(recall):
                self.log(f"train/recall_class_{i}_{class_name}", recall[i], on_step=False, on_epoch=True, batch_size=batch_size)
            if i < len(f1):
                self.log(f"train/f1_class_{i}_{class_name}", f1[i], on_step=False, on_epoch=True, batch_size=batch_size)
        # Log class support (number of samples per class in batch)
        if hasattr(self.hparams, 'label_values') and self.hparams.label_values is not None:
            unique, counts = torch.unique(labels_masked, return_counts=True)
            for u, c in zip(unique.tolist(), counts.tolist()):
                self.log(f"train/support_class_{u}", c, on_step=False, on_epoch=True, batch_size=batch_size)
        # Log number of masked/ignored points
        num_masked = mask.sum().item()
        num_total = labels.numel()
        self.log("train/num_masked", num_masked, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/num_total", num_total, on_step=False, on_epoch=True, batch_size=batch_size)
        # Log current learning rate
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"train/lr_group_{i}", param_group['lr'], on_step=True, on_epoch=False, batch_size=batch_size)
        self.log("train/confmat_mean", confmat.float().mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        preds = outputs.argmax(dim=1)
        labels = batch.labels
        loss = self.model.loss(outputs, labels)
        acc = self.model.accuracy(outputs, labels)
        mask = (labels >= 0) & (labels < self.num_classes)
        preds_masked = preds[mask]
        labels_masked = labels[mask]
        miou = self.val_miou(preds_masked, labels_masked)
        precision = self.val_precision(preds_masked, labels_masked)
        recall = self.val_recall(preds_masked, labels_masked)
        f1 = self.val_f1(preds_masked, labels_masked)
        confmat = self.val_confmat(preds_masked, labels_masked)
        batch_size = batch.features.shape[0] if hasattr(batch, 'features') else None
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/miou", miou, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        val_class_iou = self.val_class_iou(preds_masked, labels_masked)
        num_labels = len(self.hparams.label_values) if hasattr(self.hparams, 'label_values') and self.hparams.label_values is not None else self.num_classes
        for i in range(num_labels):
            if i < len(val_class_iou):
                self.log(f"val/class_iou_{i}", val_class_iou[i], on_step=False, on_epoch=True, batch_size=batch_size)

        def get_class_name(i):
            if hasattr(self.hparams, 'label_names') and self.hparams.label_names is not None:
                if i < len(self.hparams.label_names):
                    return str(self.hparams.label_names[i])
            if hasattr(self.hparams, 'label_values') and self.hparams.label_values is not None:
                if i < len(self.hparams.label_values):
                    return str(self.hparams.label_values[i])
            return str(i)
        
        for i in range(num_labels):
            class_name = get_class_name(i)
            if i < len(precision):
                self.log(f"val/precision_class_{i}_{class_name}", precision[i], on_step=False, on_epoch=True, batch_size=batch_size)
            if i < len(recall):
                self.log(f"val/recall_class_{i}_{class_name}", recall[i], on_step=False, on_epoch=True, batch_size=batch_size)
            if i < len(f1):
                self.log(f"val/f1_class_{i}_{class_name}", f1[i], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/confmat_mean", confmat.float().mean(), on_step=False, on_epoch=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        preds = outputs.argmax(dim=1)
        labels = batch.labels
        loss = self.model.loss(outputs, labels)
        acc = self.model.accuracy(outputs, labels)
        mask = (labels >= 0) & (labels < self.num_classes)
        preds_masked = preds[mask]
        labels_masked = labels[mask]
        miou = self.test_miou(preds_masked, labels_masked)
        precision = self.test_precision(preds_masked, labels_masked)
        recall = self.test_recall(preds_masked, labels_masked)
        f1 = self.test_f1(preds_masked, labels_masked)
        confmat = self.test_confmat(preds_masked, labels_masked)
        batch_size = batch.features.shape[0] if hasattr(batch, 'features') else None
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/miou", miou, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        def get_class_name(i):
            if hasattr(self.hparams, 'label_names') and self.hparams.label_names is not None:
                if i < len(self.hparams.label_names):
                    return str(self.hparams.label_names[i])
            if hasattr(self.hparams, 'label_values') and self.hparams.label_values is not None:
                if i < len(self.hparams.label_values):
                    return str(self.hparams.label_values[i])
            return str(i)
        num_labels = len(self.hparams.label_values) if hasattr(self.hparams, 'label_values') and self.hparams.label_values is not None else self.num_classes
        for i in range(num_labels):
            class_name = get_class_name(i)
            if i < len(precision):
                self.log(f"test/precision_class_{i}_{class_name}", precision[i], on_step=False, on_epoch=True, batch_size=batch_size)
            if i < len(recall):
                self.log(f"test/recall_class_{i}_{class_name}", recall[i], on_step=False, on_epoch=True, batch_size=batch_size)
            if i < len(f1):
                self.log(f"test/f1_class_{i}_{class_name}", f1[i], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test/confmat_mean", confmat.float().mean(), on_step=False, on_epoch=True, batch_size=batch_size)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=getattr(self.config, "learning_rate", 1e-2))
                # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in self.model.named_parameters() if 'offset' in k]
        other_params = [v for k, v in self.model.named_parameters() if 'offset' not in k]
        deform_lr = self.config.learning_rate * self.config.deform_lr_factor
        optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=self.config.learning_rate,
                                         momentum=self.config.momentum,
                                         weight_decay=self.config.weight_decay)
        return optimizer

    def setup(self, stage: Optional[str] = None):
        pass

    def on_train_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def forward(self, batch):
        return self.model(batch, self.config)

