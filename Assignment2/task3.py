import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
from task2a import find_mean_std

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    # Calculate the mean and standard dev from the training set
    mu, std = find_mean_std(X_train)
    # Do the pre-processing with normalisation coefficients calculated from trainset
    X_train = pre_process_images(X_train, mu, std)
    X_val = pre_process_images(X_val, mu, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Comparing the two networks with one 64 wide hidden layers and one with 10 64 wide hidden layers
    # for runtime reasons the deeper network is only trained with 5 epochs.

    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    num_epochs = 50
    learning_rate = 0.002

    model_ten_layers = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_ten_layers, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)
    shuffle_data = True

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 3c model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 4e model", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history["accuracy"], "Task 3c model")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 4e model")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
