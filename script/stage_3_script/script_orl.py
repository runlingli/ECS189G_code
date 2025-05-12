from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

#---- CNN training script for ORL ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('ORL', 'ORL dataset')  # specify the dataset name
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_CNN_ORL('CNN for ORL', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../data/stage_3_result/ORL_CNN/'
    result_obj.result_destination_file_name = 'orl_prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

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
    
    

    