'''
Concrete Evaluate class for multiple evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch

class Evaluate_Metrics(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        if isinstance(true_y, torch.Tensor):
            true_y = true_y.detach().cpu().numpy()
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.detach().cpu().numpy()
        
        accuracy = accuracy_score(true_y, pred_y)
        f1_weighted = f1_score(true_y, pred_y, average='weighted')
        
        recall_weighted = recall_score(true_y, pred_y, average='weighted')
        
        precision_weighted = precision_score(true_y, pred_y, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'recall_weighted': recall_weighted,
            'precision_weighted': precision_weighted
        } 