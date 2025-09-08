from models.model import Model
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassF1Score

"""
Model used for the single label classification tasks.
"""


class SingleLabelClassificationModel(Model):
    def __init__(self, num_classes, model, train_loader, val_loader, test_loader, backbone_layers):
        """Initializes a model for a single label classification task.

        Args:
            num_classes (int): The number of classes to predict.
            model (torch model): The specific model architecture
            train_loader (Dataloader): A pytorch dataloder for the train data.
            val_loader (Dataloader): A pytorch dataloder for the validation data.
            test_loader (Dataloader): A pytorch dataloder for the test data.
        """
        super(SingleLabelClassificationModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

        # Prepped a Resnet18 for single label classificaiton for Cifar10.
        # The resulting model is split into head and backbone.
        # Layer adjustment inspired by on a Pytorch Lightning Baseline: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )

        # Backbone
        model.maxpool = nn.Identity()
        self.backbone, self.head = self._create_backbone_and_head(model=model, backbone_layers=backbone_layers, num_classes=num_classes)

        # Hyperparameters inspired by a Pytorch Lightning Baseline: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.01,  # As no learning rate scheduling will be used, the learning rate has been decreased from the guide.
            momentum=0.9,
            weight_decay=1e-3,  # With the value from the forum post the model was overfitting too much, likely caused by the reduced data each client has. So the weight decay has been increased!
        )

        # Move the model to the appropriate device
        self = self.to(self.device)

        self.at2fl_enabled = False
    def forward(self, x):
        """Forward function of the model

        Args:
            x (torch.Tensor): Input Data

        Returns:
            torch.Tensor: Output tensor. Contains the raw scores (logits) for each class.
        """
        x = self.backbone(x)
        assert not torch.isnan(x).any(), "NaN in backbone!"
        x = self.head(x)
        return x

    def train_model(self, num_epochs, start_epoch):
        """Main model training function

        Args:
            num_epochs (int): The amoung of epochs the model shall be trained for.
            start_epoch (int): The starting epoch. This is required as after a communication round, the epoch counter shall not be reset.

        Returns:
            [dicts]: A list of dictionaries containing the training and validation metrics for each epoch.
        """

        # Set model into training mode
        self.train()
        results = []

        # Initialize metrics
        precision_metric = MulticlassPrecision(num_classes=self.num_classes).to(
            self.device
        )
        recall_metric = MulticlassRecall(num_classes=self.num_classes).to(self.device)
        f1_metric = MulticlassF1Score(num_classes=self.num_classes).to(self.device)

        # Perform training on the train data.
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(
                self.train_loader, desc="Training", unit="batch"
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()

                # clip_grad_norm is used to limit the gradients of the model's parameters during training.
                # This is a common technique to help prevent the exploding gradient problem and stabilize training
                # More information about gradient clipping: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                f1_metric.update(predicted, labels)

            # ---------- End of one training epoch ----------
            # Immediately sanitize model after AT2FL (especially BN.running_var)
            if self.at2fl_enabled:
                self._sanitize_model()

            # After processing all batches, compute the final metrics for the epoch
            precision = precision_metric.compute()
            recall = recall_metric.compute()
            f1 = f1_metric.compute()

            loss = running_loss / len(self.train_loader.dataset)
            # Print the metrics
            print(
                f"Epoch [{epoch + 1}], Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
            )

            # ---------- Validate after each epoch ----------
            # If AT2FL poisoning is enabled, switch to eval mode to stabilize batch statistics
            if self.at2fl_enabled:
                self.eval()

            # Validate after each epoch
            v_loss, v_precision, v_recall, v_f1 = self.validate_model()

            results.append(
                {
                    "epoch": epoch,
                    "train": {
                        "loss": loss,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    },
                    "val": {
                        "loss": v_loss,
                        "precision": v_precision,
                        "recall": v_recall,
                        "f1": v_f1,
                    },
                }
            )

            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()

        return results

    def validate_model(self):
        """The main validation function.

        Returns:
            [float, float, float, float]: Returns the epoch_loss, precision, recall and f1 metrics.
        """

        # Set the model into validation mode.
        self.eval()
        running_loss = 0.0

        # Initialize metrics
        precision_metric = MulticlassPrecision(num_classes=self.num_classes).to(
            self.device
        )
        recall_metric = MulticlassRecall(num_classes=self.num_classes).to(self.device)
        f1_metric = MulticlassF1Score(num_classes=self.num_classes).to(self.device)

        # Perform validation on the validation data.
        for inputs, labels in tqdm(self.val_loader, desc="Validation", unit="batch"):
            with torch.no_grad():
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                running_loss += loss.item() * inputs.size(0)

                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                f1_metric.update(preds, labels)

        precision = precision_metric.compute()
        recall = recall_metric.compute()
        f1 = f1_metric.compute()

        epoch_loss = running_loss / len(self.val_loader.dataset)

        print(
            f"Validation Loss: {epoch_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )
        return epoch_loss, precision, recall, f1

    def test_model(self):
        """The main test function.

        Returns:
            dict: A dictionary containing the epoch_loss, precision, recall and f1 metrics.
        """

        # Set model into validaiton mode.
        self.eval()
        running_loss = 0.0

        # Initialize metrics.
        precision_metric = MulticlassPrecision(num_classes=self.num_classes).to(
            self.device
        )
        recall_metric = MulticlassRecall(num_classes=self.num_classes).to(self.device)
        f1_metric = MulticlassF1Score(num_classes=self.num_classes).to(self.device)

        # Perform testing on the test data.
        for inputs, labels in tqdm(self.test_loader, desc="Testing", unit="batch"):
            with torch.no_grad():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                running_loss += loss.item() * inputs.size(0)
                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                f1_metric.update(preds, labels)

        precision = precision_metric.compute()
        recall = recall_metric.compute()
        f1 = f1_metric.compute()

        epoch_loss = running_loss / len(self.test_loader.dataset)

        test_metric = {
            "loss": epoch_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        print(
            f"Test Loss: {epoch_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )
        return test_metric

    def _sanitize_model(self):
        """
        Remove NaN/Inf from all parameters & buffers, clip weights, and
        ensure BatchNorm running statistics are valid.
        Prints what it fixed for debugging.
        """
        with torch.no_grad():
            # 1) Sanitize model parameters
            for name, p in self.named_parameters():
                n_bad = torch.isnan(p).sum() + torch.isinf(p).sum()
                if n_bad > 0:
                    print(f"[sanitize] param {name}: fixed {int(n_bad)} bad values")
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=10.0, neginf=-10.0)
                p.data.clamp_(-10.0, 10.0)

            # 2) Sanitize BatchNorm running stats
            for mod_name, m in self.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    rv_bad = torch.isnan(m.running_var).sum() + torch.isinf(m.running_var).sum()
                    rm_bad = torch.isnan(m.running_mean).sum() + torch.isinf(m.running_mean).sum()
                    if rv_bad + rm_bad > 0:
                        print(f"[sanitize] BN {mod_name}: fixed var={int(rv_bad)}, mean={int(rm_bad)}")
                    m.running_var.data  = torch.nan_to_num(m.running_var.data,  nan=1.0, posinf=1.0, neginf=1.0)
                    m.running_var.data.clamp_(min=1e-5)
                    m.running_mean.data = torch.nan_to_num(m.running_mean.data, nan=0.0, posinf=0.0, neginf=0.0)

            # 3) Sanitize non-BatchNorm buffers
            for bname, buf in self.named_buffers():
                if "running_" in bname:
                    continue
                n_bad = torch.isnan(buf).sum() + torch.isinf(buf).sum()
                if n_bad > 0:
                    print(f"[sanitize] buffer {bname}: fixed {int(n_bad)} bad values")
                    buf.data = torch.nan_to_num(buf.data, nan=0.0, posinf=0.0, neginf=0.0)
