from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
from time import time
from paths import MODELS_PATH

def train(model, train_batches, test_batches, optimizer, criterion, epochs, init_patience):
    """
    :param model: a deep model
    :param train_batches: the batches that will be used for training
    :param test_batches:the batches that will be used for testing
    :param optimizer: the optimization algorithm that used for training
    :param criterion: the loss function (almost always Binary Cross Entropy)
    :param epochs: the max number of epochs
    :param init_patience: the number of epochs that the training will last after
    its best performance in validation data
    """
    best_auc = 0
    patience = init_patience
    for i in range(1, epochs+1):
        start = time()
        val_auc = run_epoch(model, train_batches, test_batches, optimizer, criterion)
        end = time()
        print('epoch %d, auc: %2.3f. Time: %d minutes, %d seconds' % (i, 100 * val_auc, (end - start) /60, (end - start) % 60))
        if best_auc < val_auc:
            best_auc = val_auc
            patience = init_patience
            save_model(model)
            if i > 1:
                print('best epoch so far')
        else:
            patience -= 1
        if patience == 0:
            break


def run_epoch(model, train_batches, test_batches, optimizer, criterion):
    model.train(True)
    perm = np.random.permutation(len(train_batches))
    for i in perm:
        batch = train_batches[i]
        inner_perm = np.random.permutation(len(batch['text']))
        data = []
        for inp in model.input_list:
            data.append(Variable(torch.from_numpy(batch[inp][inner_perm])))
        labels = Variable(torch.from_numpy(batch['label'][inner_perm]))
        outputs = model(*data)
        loss = criterion(outputs.view(-1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, test_batches)


def evaluate(model, test_batches):
    model.train(False)
    scores_list = []
    labels_list = []
    for batch in test_batches:
        data = []
        for inp in model.input_list:
            data.append(Variable(torch.from_numpy(batch[inp])))
        outputs = model(*data)
        outputs = F.sigmoid(outputs)
        labels_list.extend(batch['label'].tolist())
        scores_list.extend(outputs.data.view(-1).tolist())

    return roc_auc_score(np.asarray(labels_list, dtype='float32'), np.asarray(scores_list, dtype='float32'))



def save_model(model):
    torch.save(model.state_dict(), MODELS_PATH+ model.name + ".mdl")


def load_model(model):
    model.load_state_dict(torch.load(MODELS_PATH+ model.name + ".mdl"))