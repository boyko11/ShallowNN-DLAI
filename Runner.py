import data_service, plotting_service, shallow_nn_utils
from shallow_nn_model import nn_model, predict
import matplotlib.pyplot as plt
import sklearn


X, Y = data_service.load_planar_dataset()

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()

# See how regular Logistic Regression does
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

plotting_service.plot_decision_boundary(lambda x: clf.predict(x), X, Y, title="Logistic Regression")

LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: ', shallow_nn_utils.accuracy(LR_predictions.reshape((1, Y.shape[1])), Y))

parameters = nn_model(X, Y, n_h=64, num_iterations=8000, print_cost=True)

# Plot the decision boundary
plotting_service.plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y,
                                        title="Decision Boundary for hidden layer size 4")

predictions = predict(parameters, X)
print('Accuracy: ', shallow_nn_utils.accuracy(predictions, Y))