import pathlib
import matplotlib.pyplot as plt
import utils
import torch.nn.functional as F
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 128  # Set number of filters in first conv layer
          # ,nn.BatchNorm2d(num_filters)
          # ,nn.ReLU()

        self.conv_layers = nn.Sequential(
           nn.Conv2d(image_channels, num_filters, kernel_size=11,stride=1,padding=2)
          ,nn.ReLU()
          ,nn.MaxPool2d(3, 2)
          ,nn.BatchNorm2d(num_filters)
          ,nn.Conv2d(num_filters, num_filters*2, kernel_size=5,stride=1,padding=2)
          ,nn.ReLU()
          ,nn.MaxPool2d(3, 2)
          ,nn.BatchNorm2d(num_filters*2)
          ,nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3,stride=1,padding=1)
          ,nn.ReLU()
          ,nn.MaxPool2d(2, 2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(2048, 100)
            ,nn.ReLU()
            ,nn.Linear(100, 10)
        )


        # TODO: Implement this function (Task  2a)
        self.num_classes = num_classes
        # Define the convolutional layers


        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        # Convolutional layers and maxpooling
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear layer
        x = self.linear_layers(x)

        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 10
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")
    # Calculate validation loss and accuracy
    validation_loss, validation_acc = compute_loss_and_accuracy(
        trainer.dataloader_val, trainer.model, trainer.loss_criterion
    )
    # Calculate training loss and accuracy
    train_loss, train_acc = compute_loss_and_accuracy(
        trainer.dataloader_train, trainer.model, trainer.loss_criterion
    )
    # Calculate test loss and accuracy
    test_loss, test_acc = compute_loss_and_accuracy(
        trainer.dataloader_test, trainer.model, trainer.loss_criterion
    )
    print('Training accuracy and loss was:',train_acc,' and ', train_loss)
    print('Validation accuracy and loss was:',validation_acc,' and ', validation_loss)
    print('Test accuracy and loss was:', test_acc, ' and ', test_loss)