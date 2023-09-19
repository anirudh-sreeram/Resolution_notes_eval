import pandas as pd
import os
import datetime

data_to_convert_list = ["OUTPUTS/9_15_2023/inputs_P24_outputs.json"]

for data_to_convert in data_to_convert_list:
    name = data_to_convert.split('/')[-1].split('.json')[0]
    df = pd.read_json(data_to_convert)

    dt_now = datetime.datetime.now()
    folder_name = '{}_{}_{}'.format(dt_now.month, dt_now.day, dt_now.year)
    path = 'EXCEL_output/{}'.format(folder_name)
    
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_excel(path + '/' + name + ".xlsx")