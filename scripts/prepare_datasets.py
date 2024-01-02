import pandas as pd 
import os
BASE_DIR = r"Data"
excel_file_path = r"Data\ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
annotated_data = pd.read_excel(excel_file_path)
csv_data = annotated_data.to_csv(os.path.join(BASE_DIR, "Data.csv"), index=False)

left_data = annotated_data[['Left-Fundus', 'Left-Diagnostic Keywords']].copy()
left_data.columns = ['Image', 'Labels']

right_data = annotated_data[['Right-Fundus', 'Right-Diagnostic Keywords']].copy()
right_data.columns = ['Image', 'Labels']

merged_data = pd.concat([left_data, right_data], index= False)
merged_data.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)



