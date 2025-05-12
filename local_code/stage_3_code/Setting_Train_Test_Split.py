'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import torch

class Setting_Train_Test_Split(setting):
    
    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)
        self.dataset = None
        self.method = None
        self.result = None
        self.evaluate = None

    def prepare(self, dataset, method, result, evaluate):
        self.dataset = dataset
        self.method = method
        self.result = result
        self.evaluate = evaluate
        
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()
        
        # run method
        self.method.data = loaded_data
        results = self.method.run()
            
        # save results
        self.result.data = results
        self.result.save()
            
        # evaluate results
        self.evaluate.data = results
        scores = self.evaluate.evaluate()
            
        return scores
