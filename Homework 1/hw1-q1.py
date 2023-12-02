#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        y_predicted = np.argmax(self.W.dot(x_i))
        if y_predicted != y_i:
            # Perceptron update
            self.W[y_i, :] +=  x_i
            self. W[y_predicted, :] -= x_i
        #raise NotImplementedError


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate):#0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        # Get probability scores according to the model (num_labels x 1).
        label_scores = np.expand_dims(self.W.dot(x_i), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_i] = 1

        # Softmax function
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # SGD update. W is num_labels x num_features.
        self.W = self.W + learning_rate * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)
        #raise NotImplementedError


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        #the number of hidden units in the hidden layer was set to 200 as per the question 1.2.2 information
        self.hidden = []
        units = [n_features, 200, n_classes]
        W1 = np.random.normal(loc=0.1,scale=0.01,size=(units[1], units[0]))
        W2 = np.random.normal(loc=0.1,scale=0.01,size=(units[2], units[1]))
        
        b1 = np.zeros((units[1],1))
        b2 = np.zeros((units[2],1))

        self.weights = [W1, W2]
        self.biases = [b1, b2]        
        #raise NotImplementedError

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predicted_values = []

        for t in range((np.shape(X)[0])):
            for i in range(len(self.weights)):
             if i == 0:
                z_input = np.dot(self.weights[0],X[t,:])+self.biases[0]
                h_input =  np.maximum(z_input, np.zeros(np.shape(z_input)[0],1))
             else: 
                z_output = np.dot(self.weights[1],h_input)+self.biases[1]
            predicted_values.append(z_output.argmax(axis=0))

        predicted_values = np.reshape(np.array(predicted_values), np.shape(np.shape(X)[0],))
        return predicted_values

        #raise NotImplementedError

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        loss = []
        for t in range((np.shape(X)[0])): #per sample
            #Foward Network

            for i in range(len(self.weights)):
             if i == 0:
                z_input = np.dot(self.weights[0],X[t,:])+self.biases[0]
                h_input =  np.maximum(z_input, np.zeros(np.shape(z_input)[0],1))
             else: 
                z_output = np.dot(self.weights[1],h_input)+self.biases[1]

            probs = np.exp(z_output) / np.sum(np.exp(z_output))
            y_one_hot = np.zeros((np.shape(z_output)[0],1))
            y_one_hot[y] = 1
            loss.append(-y.dot(np.log(probs)))

            #Backpropgation network
            grad_z = probs - y_one_hot
            grad_weights = []
            grad_biases = []
            num_layers = len(self.weights)

            for k in range(num_layers-1, -1, -1):
        
             # Gradient of hidden parameters.
                h = X[t,:] if k == 0 else h_input
                grad_weights.append(grad_z[:, None].dot(h[:, None].T))
                grad_biases.append(grad_z)
        
                # Gradient of hidden layer below.
                grad_h = self.weights[k].T.dot(grad_z)

        # Gradient of hidden layer below before activation.
                grad_z = grad_h * (1-h**2)   # Grad of loss wrt z3.

        # Making gradient vectors have the correct order
            grad_weights.reverse()
            grad_biases.reverse()

        for m in range(num_layers):
            self.weights[m] -= learning_rate*grad_weights[m]
            self.biases[m] -= learning_rate*grad_biases[m]

        return np.average(loss)


        #raise NotImplementedError


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
