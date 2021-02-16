import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # Calculates accuracy
    outputs = model.forward(X)
    accuracy = np.mean(np.sum(targets*outputs, axis = -1) - outputs.max(axis = -1) >= 0.0)
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """

        # Almost the same as previous task, calculates the cross entropy loss for multiple classes using the softmax loss equation provided in the assignment.
        targets = Y_batch
        outputs = self.model.forward(X_batch)
        self.model.backward(X_batch, outputs, targets)
        
        self.model.w += -self.learning_rate*self.model.grad
        
        loss = cross_entropy_loss(targets, outputs)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val

def plot_two_weights(model1, model2, save_as = None):
    fig, ax = plt.subplots(2, 10, gridspec_kw = {'wspace':w0, 'hspace':0,'bottom':0.67})
    for i in range(10):
        ax[0,i].imshow(model1.w[1:,i].reshape((28,28)))
        ax[0,i].axis("off")
        ax[1,i].imshow(model2.w[1:,i].reshape((28,28)))
        ax[1,i].axis("off")
    if not save_as is None:
        plt.savefig(save_as + ".png")
    plt.show()
    
    
if __name__ == "__main__":
    
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0.0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.85, .95])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    
    
    

    # Plotting of softmax weights (Task 4b)
    plot_two_weights(model, model1, "Task_4b_output")
    
    
    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1,.1, .01, .001]
    norm_vec = [np.linalg.norm(model1.w, 2)]
    acc_vec = [val_history_reg01["accuracy"]]
    
    for lbda in l2_lambdas[1:]:
        model1 = SoftmaxModel(l2_reg_lambda=lbda)
        trainer = SoftmaxTrainer(
            model1, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
        norm_vec.append(np.linalg.norm(model1.w, 2))
        acc_vec.append(val_history_reg01["accuracy"])
        
    
    for i in range(len(acc_vec)):
        x,y = zip(*sorted(acc_vec[i].items()))
        plt.plot(x,y, label = f"lambda = {l2_lambdas[i]}")

    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.bar([0,1,2,3],norm_vec)
    plt.xticks([0,1,2,3], [f"{l2_lambdas[i]}" for i in range(len(l2_lambdas))])
    plt.ylabel("L2-Norm of final Weights")
    plt.xlabel("Lambda Values")
    plt.show()
    plt.savefig("task4e_l2_reg_norms.png")
