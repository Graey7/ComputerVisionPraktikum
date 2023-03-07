import torch 
from torchmetrics.classification import BinaryAccuracy
import numpy as np
from tqdm import tqdm
from xception import xception

def make_train_step(model, loss_fn, optimizer, accuracy_fn):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)

        # METRICS
        accuracy = accuracy_fn(yhat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), accuracy.item()
    return train_step

def make_valid_step(model, loss_fn, accuracy_fn):
    def valid_step(x, y):
        model.eval()
        yhat = model(x)
        loss = loss_fn(yhat, y)

        # METRICS
        accuracy = accuracy_fn(yhat, y)

        return loss.item(), accuracy.item()
    return valid_step

def train(train_data, valid_data, epochs, device):
    # INIT MODEL
    model = xception(output='softmax', pretrained=True)

    # INIT LOSS
    loss_fn = torch.nn.BCELoss()

    # INIT OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    # INIT SCHEDULER 
    # TODO: CHECK CORRECT IMPLEMENTATION FOR SGD WITH RESTARTS
    ''' https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup as alternative? '''
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=0.3, eta_mmin=1e-5)
     
    # INIT METRICS
    accuracy_metric = BinaryAccuracy()
    accuracy_metric = accuracy_metric.to(device)

    train_step = make_train_step(model, loss_fn, optimizer, accuracy_metric)
    valid_step = make_valid_step(model, loss_fn, accuracy_metric)

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    # TRAIN LOOP
    for epoch in tqdm(range(epochs), desc='epoch'):
        
        # TRAIN STEP
        tmp_train_loss = []
        tmp_train_acc = []
        for x_batch, y_batch in train_data:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss, accuracy = train_step(x_batch, y_batch)

            tmp_train_loss.append(loss)
            tmp_train_acc.append(accuracy)

        train_loss.append(np.mean(tmp_train_loss))
        train_acc.append(np.mean(tmp_train_acc))

        # VALIDATION STEP
        tmp_valid_loss = []
        tmp_valid_acc = []
        with torch.no_grad():
            for x_val, y_val in valid_data:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                loss, accuracy = valid_step(x_val, y_val)

                tmp_valid_loss.append(loss)
                tmp_valid_acc.append(accuracy)

            valid_loss.append(np.mean(tmp_valid_loss))
            valid_acc.append(np.mean(tmp_valid_acc))

        scheduler.step()

    return model, train_loss, valid_loss, train_acc, valid_acc

def make_test_step(model, accuracy_fn):
    def test_step(x, y):
        model.eval()
        yhat = model(x)

        # METRICS
        accuracy = accuracy_fn(yhat, y)

        return accuracy.item()
    return test_step

def test(model, test_data, device):
    accuracy_metric = BinaryAccuracy()
    accuracy_metric = accuracy_metric.to(device)

    test_acc = []

    test_step = make_test_step(model, accuracy_metric)

    with torch.no_grad():
        for x_test, y_test in test_data:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            accuracy = test_step(x_test, y_test)

            test_acc.append(accuracy)

    return np.mean(test_acc) 