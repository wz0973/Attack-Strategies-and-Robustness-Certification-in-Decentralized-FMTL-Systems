from models.model import Model
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

"""
Model used for the face landmark point detection tasks.
"""


class LandmarkDetectionModel(Model):
    def __init__(self, num_landmarks, model, train_loader, val_loader, test_loader, backbone_layers):
        """Initializes a model for a face landmark point detection task.

        Args:
            num_classes (int): The number of classes to predict.
            model (torch model): The specific model architecture
            train_loader (Dataloader): A pytorch dataloder for the train data.
            val_loader (Dataloader): A pytorch dataloder for the validation data.
            test_loader (Dataloader): A pytorch dataloder for the test data.
        """
        super(LandmarkDetectionModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_landmarks = (
            num_landmarks * 2
        )  # Because each landmark has x and y coordinates

        # Preparing a Resnet18 model for point detection.
        self.backbone, self.head = self._create_backbone_and_head(model=model, backbone_layers=backbone_layers, num_classes=self.num_landmarks)

        # hyperparameters. Values were chosen by non-rigourous hyperparameter tuning.
        self.criterion = (
            nn.MSELoss()
        )  # MSELoss because point detection is a regression task
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        # Move the model to the appropriate device
        self = self.to(self.device)

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
        """
        Main model training function.

        Workflow:
            for epoch in range(...):
                for batch in train_loader:
                    input → model → output
                    compute MSE loss
                    backpropagate loss
                    clip gradients (to avoid explosion)
                    update model parameters using optimizer
                validate model on val_loader
                log validation loss
                apply learning rate scheduler (decay)
        """

        # Set model into training mode
        self.train()
        train_losses = []
        val_losses = []
        results = []

        # ---
        # This code snippet was used to see the impact of using Batch Normalization (https://en.wikipedia.org/wiki/Batch_normalization).
        # To activate set use_batchNorm to True.
        use_batchNorm = False
        if start_epoch > 0 and use_batchNorm:
            for _ in range(3):  # Repeat 3 times
                with torch.no_grad():  # Disable gradient calculation
                    for inputs, _ in tqdm(
                        self.train_loader,
                        desc=f"Updating BatchNorm {_}/3",
                        unit="batch",
                    ):
                        inputs = inputs.to(self.device)
                        _ = self(inputs)  # Forward pass to update BatchNorm statistics

        # ---

        # Perform training on the train data.
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in tqdm(
                self.train_loader, desc="Training", unit="batch"
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # clip_grad_norm is used to limit the gradients of the model's parameters during training.
                # This is a common technique to help prevent the exploding gradient problem and stabilize training
                # More information about gradient clipping: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            # Calculate average loss for the epoch
            avg_train_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(avg_train_loss)

            # Print the final metrics
            print(f"Epoch [{epoch + 1}], Loss: {avg_train_loss:.4f}")

            # Validate after each epoch
            val_loss = self.validate_model()
            val_losses.append(val_loss)
            self.scheduler.step()

            results.append(
                {
                    "epoch": epoch,
                    "train": {"loss": loss},
                    "val": {"loss": val_loss},
                }
            )

        return results

    def validate_model(self):
        """The main validation function.

        Returns:
            float: Returns the validation loss.
        """

        # Set the model into validation mode.
        running_loss = 0.0
        self.eval()

        # Perform validation on the validation data.
        with torch.no_grad():
            for inputs, labels in tqdm(
                self.val_loader, desc="Validation", unit="batch"
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        avg_val_loss = running_loss / len(self.val_loader.dataset)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def test_model(self):
        """The main test function.

        Returns:
            float: Returns the test loss.
        """

        # Set the model into validation mode.
        self.eval()
        running_loss = 0.0
        self.to(self.device)

        # Perform validation on the test data.
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Testing", unit="batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.test_loader.dataset)

        test_metric = {"loss": epoch_loss}

        print(f"Test Loss: {epoch_loss:.4f}")
        return test_metric
