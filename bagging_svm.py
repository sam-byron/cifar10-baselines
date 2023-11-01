import tensorflow as tf
from tensorflow.keras import datasets
tf.get_logger().setLevel('ERROR')
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
import time

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

X = np.reshape(train_images, (50000, 32*32*3))
y = train_labels

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Turn down for faster convergence
t0 = time.time()
train_samples = 5000*4

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# clf = svm.LinearSVC(dual=False)

n_estimators = 32
clf = BaggingClassifier(estimator=svm.LinearSVC(verbose=1, max_iter=250), n_estimators=n_estimators, n_jobs=-1, verbose=1, max_samples=5000)
clf.fit(X, y)

score = clf.score(X_test, y_test.ravel())
print("Test score: %.4f" % score)

run_time = time.time() - t0
print("Example run in %.3f s" % run_time)

# Test score: 0.4614
# Example run in 355.745 s
