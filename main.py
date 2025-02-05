import os
import numpy as np
from copy import deepcopy
import torch
import models
from clients.FCLPF import FCLPF
from models.ResNet import ResNet18
from models.myNetwork import network
from data_prep.data import CL_dataset
from utiles import setup_seed, fedavg_aggregation, evaluate_accuracy_forgetting, evaluate_accuracy, start, aggregate_prototypes, generate_prototype


args = start()

result_path = args.result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID
setup_seed(args.seed)

if args.dataset == 'CIFAR100':
    dataset = CL_dataset(args)
    feature_extractor = ResNet18(args.num_classes, cifar=True)
    ds = dataset.train_dataset
elif args.dataset == 'TinyImageNet':
    dataset = CL_dataset(args)
    feature_extractor = ResNet18(args.num_classes, cifar=False)
    ds = dataset.train_dataset
    

# Initialize global model
global_model = network(dataset.initial_classes, feature_extractor)
teacher, generator = None, None
gamma = np.log(args.lr_end / args.lr)
task_size = dataset.n_classes_per_task
counter, classes_learned = 0, dataset.initial_classes
num_participants = int(args.frac * args.num_clients)
clients, max_accuracy = [], []


for i in range(args.num_clients):
    group = dataset.groups[i]
    client_id = i
    if args.method == 'FCLPF':
        client = FCLPF(args.batch_size, args.epochs, ds, group, args.dataset, client_id=client_id, device='cuda')
        pseudo_feature_path = f'{args.pseudo_feature_base_path}/client_{client_id}'
        client.pseudo_feature_path = pseudo_feature_path
        if not os.path.exists(pseudo_feature_path):
            os.makedirs(pseudo_feature_path) 
    clients.append(client)

# Training and evaluation loop
for t in range(args.n_tasks):
    test_loader = dataset.get_full_test(t)
    [client.set_next_t() for client in clients]
        
    if t ==0:    
        # Initial task training
        for round in range(args.global_round):
            weights = []
            lr = args.lr * np.exp(round / args.global_round * gamma)
            selected_clients = [clients[idx] for idx in np.random.choice(args.num_clients, num_participants, replace=False)]
            for user in selected_clients:
                model = deepcopy(global_model)
                user.train(model, lr)
                weights.append(model.state_dict())
            global_model.load_state_dict(fedavg_aggregation(weights))
            correct, total = evaluate_accuracy(global_model, test_loader, args.method, result_path=result_path, round=round, total_round=args.global_round, task_index=t)
            print(f'round {counter}, accuracy: {100 * correct / total}')
            counter += 1

        # Prototype aggregation after initial task
        all_prototypes = []
        class_counts = []        
        for client in clients:
            features, labels = client.extract_features(global_model)
            prototype=deepcopy(generate_prototype(features, labels))
            all_prototypes.append(prototype)
            class_count = {class_id.item(): torch.sum(labels == class_id).item() for class_id in torch.unique(labels)}
            class_counts.append(class_count)
        aggregated_prototype = aggregate_prototypes(all_prototypes, class_counts)
        max_accuracy.append(correct / total)
        
        for client in clients:
            client.update_prototype(aggregated_prototype)
        
    else:
        # Subsequent tasks training
        for round in range(args.global_round):
            weights = []
            lr = args.lr * np.exp(round / args.global_round * gamma)
            selected_clients = [clients[idx] for idx in np.random.choice(args.num_clients, num_participants, replace=False)]           

            if round == 0:
                # Prototype aggregation for new task
                all_prototypes = []
                class_counts = []
                for client in clients:
                    features, labels = client.extract_features(global_model)
                    prototype=deepcopy(generate_prototype(features, labels))
                    all_prototypes.append(prototype)
                    class_count = {class_id.item(): torch.sum(labels == class_id).item() for class_id in torch.unique(labels)}
                    class_counts.append(class_count)
                aggregated_prototype = aggregate_prototypes(all_prototypes, class_counts)
                
                for client in clients:
                    client.update_prototype(aggregated_prototype)
                
                for client in clients:
                    current_features, current_labels = client.extract_features(global_model)
                    client.generate_pseudo_features( current_features, current_labels, t)
                            
            for user in selected_clients:
                current_features, current_labels = user.extract_features(global_model)
                model = deepcopy(global_model.fc) 
                user.train_mlp(model ,current_labels, current_features, t,lr=lr)
                weights.append(model.state_dict())
                
            global_model.fc.load_state_dict(fedavg_aggregation(weights))
            
            correct, total = evaluate_accuracy(global_model, test_loader, args.method, result_path=result_path, round=round, total_round=args.global_round, task_index=t )
            print(f'round {counter}, accuracy: {100 * correct / total}')
            counter += 1
        
    # Evaluate forgetting    
    correct, total, accuracies = evaluate_accuracy_forgetting(global_model, dataset.get_cl_test(t), args.method)
    print(f"total_accuracy_{t}: {accuracies}")
    max_accuracy.append(accuracies[-1])
    classes_learned += task_size
    global_model.Incremental_learning(classes_learned)
print('forgetting:', sum([max_accuracy[i] - accuracies[i] for i in range(args.n_tasks)]) / args.n_tasks)
