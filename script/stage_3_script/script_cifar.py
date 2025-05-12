from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

#---- CNN training script for CIFAR-10 ----
if __name__ == '__main__':
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('CIFAR', 'CIFAR-10 dataset')  # specify the dataset name
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN_CIFAR('CNN for CIFAR-10', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../data/stage_3_result/CIFAR_CNN/'
    result_obj.result_destination_file_name = 'cifar_prediction_result'

    setting_obj = Setting_Train_Test_Split('CIFAR', 'CNN on CIFAR-10 dataset')

    evaluate_obj = Evaluate_Metrics('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print("\nEvaluation Results:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    print('************ Finish ************')
    # ------------------------------------------------------
    

    