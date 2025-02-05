import os
import pickle
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.decomposition import PCA



class FCLPF:
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name, client_id=None, device='cuda',):
        # Initialize the FCLPF client
        self.criterion_fn = F.cross_entropy
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.groups = groups
        self.current_t = -1
        self.local_epoch = epochs
        self.dataset_name = dataset_name
        self.prototype = {}
        self.client_id = client_id
        self.pseudo_feature_path = None
        self.device=device
        

    def set_dataloader(self, samples):
        self.train_loader = DataLoader(Subset(self.train_dataset, samples), batch_size=self.batch_size, shuffle=True)


    def set_next_t(self):
        # Sets the dataloader for the next task
        self.current_t += 1
        samples = self.groups[self.current_t]
        self.set_dataloader(samples)

    def train(self, model, lr):
        # Trains the model using local data
        model.to('cuda')
        model.train()
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                logits = model(x)
                loss = self.criterion_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.to('cpu')
    
    def extract_features(self, model):
        # Extracts features from the model using local data
        model.to('cuda')
        model.eval()
        features, labels = [], []
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                features.append(model.feature_extractor(x))
                labels.append(y)
        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels
    
    
    def update_prototype(self, new_prototype):
        # Updates the prototype with new prototype values
        self.prototype.update(new_prototype)
    
        
    def generate_pseudo_features(self,  current_features, current_labels, task_index):
        # Generates pseudo features for past classes
        current_labels = current_labels.cpu() 
        current_features = current_features.cpu()
        new_class_means = {label: self.prototype[label] for label in np.unique(current_labels)}
        pseudo_features = []
        pseudo_labels = []
        old_class_labels = [label for label in self.prototype.keys() if label not in new_class_means.keys()]
        num_data_points = len(np.unique(current_labels))+1
        n_components = min(num_data_points, 512) 
        pseudo_features = []
        pseudo_labels = []

        
        for old_label in old_class_labels:
            if old_label in self.prototype:
                old_class_mean = self.prototype[old_label]

                
                closest_new_label = self.find_closest_class(new_class_means, old_class_mean)
                new_class_mean = new_class_means[closest_new_label]
                
                old_class_mean_reshaped = old_class_mean.reshape(1, -1)
                new_class_means_array = np.array([mean.reshape(1, -1) for mean in new_class_means.values()])

                # PCA
                pca_input = np.vstack([old_class_mean_reshaped] + list(new_class_means_array))
                pca = PCA(n_components=n_components)
                pca.fit(pca_input)
                
                principal_axes = pca.components_
                difference_vector = old_class_mean_reshaped - new_class_mean.reshape(1, -1)
                projected_difference = np.dot(np.dot(difference_vector, principal_axes.T), principal_axes)
                projected_difference = projected_difference.reshape(-1)

              
                for feature in current_features[current_labels == closest_new_label]:
                    pseudo_feature = feature + projected_difference
                    pseudo_features.append(pseudo_feature)
                    pseudo_labels.append(old_label)
        
        self.save_pseudo_features(pseudo_features, pseudo_labels, task_index)
        
    def find_closest_class(self, new_class_means, old_class_mean):
        # Finds the closest new class to the old class mean
        closest_label = None
        highest_similarity = -np.inf
        old_class_mean = old_class_mean.flatten()  

        for label, mean in new_class_means.items():
            mean = mean.flatten() 
            similarity = np.dot(mean, old_class_mean) / (np.linalg.norm(mean) * np.linalg.norm(old_class_mean))
            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_label = label
        return closest_label        
    
        
        
    def save_pseudo_features(self, pseudo_features, pseudo_labels, task_index):
        # Saves the pseudo features and labels to a file
        save_path = os.path.join(self.pseudo_feature_path, f'pseudo_features_task_{task_index}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump((pseudo_features, pseudo_labels), f)
    
    def load_pseudo_features(self, task_index):
        # Loads the pseudo features and labels from a file
        load_path = os.path.join(self.pseudo_feature_path, f'pseudo_features_task_{task_index}.pkl')
        with open(load_path, 'rb') as f:
            pseudo_features, pseudo_labels = pickle.load(f)
        return torch.stack(pseudo_features),  torch.tensor(pseudo_labels, dtype=torch.long)
    
    def train_mlp(self,model ,current_labels, current_features, task_index,lr):
        # Trains the MLP using current and pseudo features
        pseudo_features, pseudo_labels = self.load_pseudo_features(task_index)        
        pseudo_features = pseudo_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
       
        combined_features = torch.cat((current_features, pseudo_features), dim=0)
        combined_labels = torch.cat((current_labels, pseudo_labels), dim=0)
        
        assert len(combined_features) == len(combined_labels)
        
        
        combined_dataset = TensorDataset(combined_features.clone().detach(), combined_labels.clone().detach())
        loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
        
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        
        
        model.train()
        for epoch in range(self.local_epoch):
            for features, labels in loader:
                features, labels = features.to(self.device), labels.to(self.device)
                opt.zero_grad()
                outputs = model(features)
                loss = self.criterion_fn(outputs, labels)
                loss.backward()
                opt.step()
        
        