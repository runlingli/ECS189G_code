'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle
import os

class Result_Saver(result):
    data = None
    fold_count = 0 
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')

        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        file_path = os.path.join(
            self.result_destination_folder_path,
            f"{self.result_destination_file_name}_{self.fold_count}.pkl"
        )

        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f)

        print(f'Results saved to: {file_path}')
