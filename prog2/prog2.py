import numpy as np
import matplotlib.pyplot as plt
import argparse 



#gradient descent solution
def gradient_descent(X, y, m, b, Learning_rate, epochs):
    for i in range(epochs):
        y_pred = m * X + b
        error = y - y_pred
        m_gradient = (-2/n) * X.dot(error)
        b_gradient = (-2/n) *np.sum(error)
        m -= Learning_rate * m_gradient
        b -= Learning_rate * b_gradient
    return m, b


#closed form solution
def closed_form_solution(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]  # add bias term
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best[1], theta_best[0]  # return m and b


#prediction mode
def predict(x, m, b):
    return m * x + b

def evaluate(x, y, m, b):
    y_pred = predict(x, m, b)
    mse = np.mean((y - y_pred) ** 2)
    return mse

def main():
    parser = argparse.ArgumentParser(description ='',
    formatter_class=argparse.RawTextHelpFormatter)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train a model from x.txt and y.txt')
    mode_group.add_argument('--predict', action='store_true', help='Run prediction on a dev set using a trained model')
    mode_group.add_argument('--eval', action='store_true', help='Evaluate a trained model on a dev set (prints MSE)')
    
    parser.add_argument('--order', metavar='ORDER', type=int, default=1, help='Polynomial order (1 for linear model, default 1)')
    parser.add_argument('--x', metavar='X', type=str, help='Path to inputs x.txt.')
    parser.add_argument('--y', metavar='Y', type=str, help='Path to targets y.txt.')
    parser.add_argument('--model', metavar='MODEL', type=str, help='Path to model file to be written (train) or read (predict/eval).')
    parser.add_argument('--preds', metavar='PREDS', type=str, help='Path to write predictions (predict).')
    parser.add_argument('--lr', metavar='LR', type=float, type=float, help='Learning rate for gradient descent (train).')
    parser.add_argument('--max-steps', metavar='MAX_STEPS', type=int, default=0, help='Max gradient steps; 0 means analytical solution (train), default 0.')
