import os
from sklearn.metrics import confusion_matrix
import pickle
import torch
import random
import argparse
import numpy as np
from copy import deepcopy



def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    # torch.cuda.empty_cache()


def fedavg_aggregation(weights):
    w_avg = deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg

def save_final_results(result_path, accuracy, confusion_mtx, task_index):
    final_results_path = os.path.join(result_path, f'task_{task_index}_final_results.pkl')
    with open(final_results_path, 'wb') as f:
        pickle.dump({'accuracy': accuracy, 'confusion_matrix': confusion_mtx}, f)


def save_results(result_path, task_index,round, accuracy, confusion_mtx):
    with open(os.path.join(result_path, 'round_accuracy_results.txt'), 'a') as f:
        f.write(f'Task: {task_index}, Round: {round}, Accuracy: {accuracy}\n')
    with open(os.path.join(result_path, 'round_confusion_results.txt'), 'a') as f:
        f.write(f'Task: {task_index}, Round: {round}, Confusion Matrix:\n{confusion_mtx}\n')



def evaluate_accuracy(model, test_loader, method=None,  result_path=None, round=None, total_round=None, task_index=None):
    model.to('cuda')
    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_preds = []
    
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to('cuda'), y.to('cuda')
        with torch.no_grad():
            outputs = model(x)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == y.cpu()).sum()
        total += len(y)
        
        all_labels.extend(y.cpu().numpy())
        all_preds.extend(predicts.cpu().numpy())
        
    accuracy = 100 * correct / total
    confusion_mtx = confusion_matrix(all_labels, all_preds)
    if result_path is not None and round is not None:
        save_results(result_path, task_index,round, accuracy, confusion_mtx)
        if round == total_round-1:
            save_final_results(result_path, accuracy, confusion_mtx, task_index)
        
    model.to('cpu')
    
    
    return correct, total


def evaluate_accuracy_forgetting(model, test_loaders, method=None):
    c, t = 0, 0
    accuracies = []
    for task_id, test_loader in enumerate(test_loaders):
        ci, ti = evaluate_accuracy(model, test_loader, method)
        accuracies.append(100 * ci / ti)
        c += ci
        t += ti
    return c, t, accuracies



def combine_data(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x, y = torch.cat(x), torch.cat(y)
    return x, y

def generate_prototype(features, labels):
        unique_labels = torch.unique(labels)
        prototype = {}
        for label in unique_labels:
            class_features = features[labels == label]
            prototype[label.item()] = class_features.mean(dim=0).cpu().numpy()
        return prototype
    
def aggregate_prototypes(prototypes, class_counts):
    aggregated_prototype = {}
    total_class_counts = {class_id: 0 for class_id in class_counts[0].keys()}

    # 각 클래스별 가중합 계산
    for i, client_prototype in enumerate(prototypes):
        for class_id, prototype in client_prototype.items():
            if class_id not in aggregated_prototype:
                aggregated_prototype[class_id] = np.zeros_like(prototype)
            aggregated_prototype[class_id] += prototype * class_counts[i][class_id]
            if class_id not in total_class_counts:
                total_class_counts[class_id] = 0
            total_class_counts[class_id] += class_counts[i][class_id]
    
    for class_id in aggregated_prototype.keys():
        aggregated_prototype[class_id] /= total_class_counts[class_id]
    
    return aggregated_prototype


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuID', type=str, default='0', help="GPU ID")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--method', type=str, default='FCLPF', help="name of method", choices=['FCLPF'])
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="name of dataset")
    parser.add_argument('--num_clients', type=int, default=50, help='#clients')
    parser.add_argument('--epochs', type=int, default=10, help='Local Epoch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Local Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Local Bachsize')
    parser.add_argument('--eval_int', type=int, default=10, help='Evaluation intervals')
    parser.add_argument('--global_round', type=int, default=100, help='#global rounds per task')
    parser.add_argument('--frac', type=float, default=0.1, help='#selected clients in each round')
    parser.add_argument('--alpha', type=float, default=1, help='LDA parameter for data distribution')
    parser.add_argument('--n_tasks', type=int, default=10, help='#tasks')
    parser.add_argument('--pi', type=int, default=100, help='local epochs of each global round')
    parser.add_argument('--z_dim', type=int, default=1000)
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--w_ie', type=float, default=1.)
    parser.add_argument('--w_kd', type=float, default=1e-1)
    parser.add_argument('--w_ft', type=float, default=1)
    parser.add_argument('--w_act', type=float, default=0.1)
    parser.add_argument('--w_noise', type=float, default=1e-3)
    parser.add_argument('--w_bn', type=float, default=5e1)
    parser.add_argument('--path', type=str, help='path to dataset')
    parser.add_argument('--result_path', type=str, default='./results/cifarS2', help='Path to save results')
    parser.add_argument('--pseudo_feature_base_path', type=str, default='./data/cifarS2', help='Base path for pseudo features')
    parser.add_argument('--initial_classes', type=int, default=50, help='number of initial classes')
    parser.add_argument('--increment_classes', type=int, default=5, help='number of increment classes')
    args = parser.parse_args()
    args.lr_end = 0.01
    return args
