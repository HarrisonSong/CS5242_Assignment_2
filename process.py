from utils import datasets
from applications import SentimentNet
from loss import SoftmaxCrossEntropy, L2
from optimizers import Adam
import numpy as np
np.random.seed(5242)

dataset = datasets.Sentiment()
model = SentimentNet(dataset.dictionary)
loss = SoftmaxCrossEntropy(num_class=2)

adam = Adam(lr=0.001, decay=0,
            scheduler_func=lambda lr, it: lr*0.5 if it%1000==0 else lr)
model.compile(optimizer=adam, loss=loss, regularization=L2(w=0.001))
train_results, val_results, test_results = model.train(
        dataset,
        train_batch=20, val_batch=100, test_batch=100,
        epochs=5,
        val_intervals=100, test_intervals=300, print_intervals=5)