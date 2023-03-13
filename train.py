import torch 
from torchmetrics.classification import BinaryAccuracy
import numpy as np
from tqdm import tqdm
from xception import xception

# HELPER FUNCTIONS FOR TRAINING LOOP
# RETURN FUNCTION THAT CAN BE CALLED IN THE LOOP
def make_train_step(model, loss_fn, optimizer, accuracy_fn):
    def train_step(x, y):
        # SET MODEL TO TRAIN MODE
        model.train()
        # CALCULATE PREDICTION AND LOSS
        yhat = model(x)
        loss = loss_fn(yhat, y)

        # METRICS
        accuracy = accuracy_fn(yhat, y)

        # BACKPROPAGATION
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), accuracy.item()
    return train_step

def make_valid_step(model, loss_fn, accuracy_fn):
    def valid_step(x, y):
        # SET MODEL TO EVALUATION MODE
        model.eval()
        # CALCULATE PREDICTION AND LOSS
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

    # INIT TRAIN AND VALID STEP
    train_step = make_train_step(model, loss_fn, optimizer, accuracy_metric)
    valid_step = make_valid_step(model, loss_fn, accuracy_metric)

    # ARRAYS FOR METRICS
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

            # TRAIN MODEL
            loss, accuracy = train_step(x_batch, y_batch)

            tmp_train_loss.append(loss)
            tmp_train_acc.append(accuracy)
        
        # CALCULATE MODEL TRAIN METRICS FOR EPOCH
        train_loss.append(np.mean(tmp_train_loss))
        train_acc.append(np.mean(tmp_train_acc))

        # VALIDATION STEP
        tmp_valid_loss = []
        tmp_valid_acc = []
        with torch.no_grad():
            for x_val, y_val in valid_data:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                # VALID MODEL
                loss, accuracy = valid_step(x_val, y_val)

                tmp_valid_loss.append(loss)
                tmp_valid_acc.append(accuracy)

            # CALCULATE MODEL VALID METRICS FOR EPOCH
            valid_loss.append(np.mean(tmp_valid_loss))
            valid_acc.append(np.mean(tmp_valid_acc))

        # ADJUST LR 
        scheduler.step()

    # RETURN TRAINED MODEL AND METRICS 
    return model, train_loss, valid_loss, train_acc, valid_acc

# HELPER FUNCTIONS FOR TEST
# RETURN FUNCTION THAT CAN BE CALLED IN TEST
def make_test_step(model, accuracy_fn):
    def test_step(x, y):
        # SET MODEL TO TRAIN MODE
        model.eval()
        yhat = model(x)

        # METRICS
        accuracy = accuracy_fn(yhat, y)

        return accuracy.item()
    return test_step

def test(model, test_data, device):
    # INIT METRICS
    accuracy_metric = BinaryAccuracy()
    accuracy_metric = accuracy_metric.to(device)

    # ARRAYS FOR METRIC
    test_acc = []
    
    # INIT TEST STEP
    test_step = make_test_step(model, accuracy_metric)

    # TEST STEP
    with torch.no_grad():
        for x_test, y_test in test_data:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # TEST MODEL
            accuracy = test_step(x_test, y_test)

            # CALCULATE MODEL PERFORMANCE ON TESTSET
            test_acc.append(accuracy)

    # RETURN ACCURACY ON TESTSET
    return np.mean(test_acc) 