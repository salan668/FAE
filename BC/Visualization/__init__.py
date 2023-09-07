def LegendRename(name_list):
    rename_dict= {'cv_train': 'CV Training', 'cv_val': 'Validation',
                  'balance_train': 'Balance Training',
                  'train': 'Training', 'test': 'Test'}
    new_name_list = [rename_dict[i] for i in name_list]
    return new_name_list