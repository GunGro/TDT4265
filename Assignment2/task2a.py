import numpy as np
import utils
import typing

np.random.seed(1)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def improved_sigmoid(X):
    return 1.7159*np.tanh((2.0/3)*X) #np.tanh(X)#

def dsigmoid(Z):
    return Z*(1-Z)

def improved_dsigmoid(Z):
    return (2.0/3)*(1.7159-Z**2/1.7159) #1-Z**2#

def pre_process_images(X: np.ndarray, mu, std):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    X = np.array(X, dtype = np.float64)
    X = (X - mu)/std
    X = np.append(X, np.ones((X.shape[0], 1)), axis = 1) #append a one to the end
    return X


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # Implements on hot encoded vectors.
    vec = np.zeros((Y.shape[0],num_classes))
    vec[np.arange(Y.shape[0]), np.array(Y[:,0], dtype = np.int)] = 1.0
    return vec

def find_mean_std(X: np.ndarray):

    # Takes inn an array which should be the whole training set and returns mu and std
    std = np.std(X)
    mu = np.mean(X)
    return mu, std

def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    assert targets.shape == outputs.shape, \
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    cost = -np.mean((np.sum(targets * np.log(outputs), axis=-1)))
    return cost


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        if self.use_improved_sigmoid:
            self.act = improved_sigmoid
            self.deact = improved_dsigmoid
        else:
            self.act = sigmoid
            self.deact = dsigmoid
        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer


        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size) #(size, prev) #
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                w = np.random.normal(0, 1/np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        # Cache previous outputs
        self.hidden_layer_outputs = [None]*(len(self.ws)-1)
        self.softmax_input = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_ouput = ...

        outputs = X

        for i, weights in enumerate(self.ws[:-1]):
            outputs = self.act(np.matmul(outputs, weights))
            self.hidden_layer_outputs[i] = outputs

        # Passing from final hidden layer to output layer
        outputs = np.matmul(outputs, self.ws[-1])
        # Softmaxing outputlayer
        outputs = np.exp(outputs)/(np.sum(np.exp(outputs), axis=-1, keepdims = True))
        return outputs

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        # calculate the backwards pass across the softmax


        def cal_weights_grad(dact, y, dC_dz):
            return np.mean(np.einsum("ij,ik->ijk", y, dact*dC_dz), axis=0)

        def cal_delta(dact, weights, dC_dz):
            return np.einsum("ij,kj->ik", dC_dz*dact, weights)
            

        
        delta = -(targets-outputs)
        # get gradient matrix
        self.grads[-1] = np.mean(np.einsum('ij,ik->ijk', self.hidden_layer_outputs[-1],delta),axis=0)
        # get dLoss / dlast_hidden_layer
        delta = np.einsum('ij, kj -> ik', delta, self.ws[-1])
        # do the hidden layers        
        for i, weights in enumerate(reversed(self.ws[1:-1])):
            z = self.hidden_layer_outputs[-1-i]
            dact = self.deact(z) 
            # Find the weight gradient
            self.grads[-2-i] = cal_weights_grad(dact, self.hidden_layer_outputs[-2-i], delta )
            # find the loss derivative of hidden layer y to pass backwards
            delta = cal_delta(dact, weights, delta)
        
        #do the first hidden layer
        z = self.hidden_layer_outputs[0]
        dact = self.deact(z)
        # Find the weight gradient
        self.grads[0] = cal_weights_grad(dact, X, delta )

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
#



def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    # Actual gradient
    logits = model.forward(X)
    model.backward(X, logits, Y)
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig

                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    mu, std = find_mean_std(X_train)
    X_train = pre_process_images(X_train,mu, std)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
