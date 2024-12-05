'''
Code for processing the D7P dataset

Code is adapted from directory https://github.com/ZerojumpLine/Robust-Skin-Lesion-Classification. Many thanks to Zeju Li!
'''
import os
import pandas as pd
import shutil

#Can change where data is saved, default is left blank.
D7P_save_dir = ''
D7P_downloaded_data = ''


#Donwload the Meta file
D7P_downloaded_data_Descriptions = D7P_downloaded_data + 'release_v0/meta/meta.csv'
descriptionfile = pd.read_csv(D7P_downloaded_data_Descriptions)
diagnosis = descriptionfile['diagnosis']
clinic_file = descriptionfile['clinic']
derm_file = descriptionfile['derm']


# Initialise data for train, valid, and test CSVs
train_indices_path = os.path.join(D7P_downloaded_data, 'release_v0/meta/train_indexes.csv')
valid_indices_path = os.path.join(D7P_downloaded_data, 'release_v0/meta/valid_indexes.csv')
test_indices_path = os.path.join(D7P_downloaded_data, 'release_v0/meta/test_indexes.csv')

train_indices = pd.read_csv(train_indices_path).values.flatten()
valid_indices = pd.read_csv(valid_indices_path).values.flatten()
test_indices = pd.read_csv(test_indices_path).values.flatten()

train_data = []
valid_data = []
test_data = []

#Dictionary for dermoatology classes
diagnosis_to_folder = {
    'nevus': 'nevus',
    'melanoma': 'melanoma',
    'dermatofibroma': 'dermatofibroma',
    'basal cell carcinoma': 'basal_cell_carcinoma',
    'seborrheic keratosis': 'pigmented_benign_keratosis',
    'vascular lesion': 'vascular_lesion',
}

def sort_diagnosis(diagnosis):
    # Check for exact matches in the map
    if diagnosis in diagnosis_to_folder:
        return diagnosis_to_folder[diagnosis]
    
    # Check partial matches (e.g., endswith or startswith)
    if diagnosis.endswith('nevus'):
        return 'nevus'
    if diagnosis.startswith('melanoma'):
        return 'melanoma'
    
    return None  # Default if no match is found
    

# Pre-create all required directories
for folder in set(diagnosis_to_folder.values()):
    outputfolder = os.path.join(D7P_save_dir, folder)
    os.makedirs(outputfolder, exist_ok=True)


#Sort through data
for count in range(len(diagnosis)):
        folder_name = sort_diagnosis(diagnosis[count])
        if folder_name != None:  # Check if diagnosis is in the mapping
            outputfolder = os.path.join(D7P_save_dir, folder_name)

            image_name_clinic = clinic_file[count].split('/')[-1] if pd.notnull(clinic_file[count]) else None
            image_name_derm = derm_file[count].split('/')[-1] if pd.notnull(derm_file[count]) else None
            one_hot = {key: 0 for key in diagnosis_to_folder.values()}
            one_hot[folder_name] = 1
            
            # Copy 'clinic' images
            if pd.notnull(clinic_file[count]):
                src1 = os.path.join(D7P_downloaded_data, 'release_v0/images', clinic_file[count])
                dst1 = os.path.join(outputfolder, clinic_file[count][4:])
                if os.path.isfile(src1):
                    shutil.copyfile(src1, dst1)

                row = [image_name_clinic, f"{folder_name}/{image_name_clinic}"] + list(one_hot.values())
                if count in train_indices:
                    train_data.append(row)
                elif count in valid_indices:
                    valid_data.append(row)
                elif count in test_indices:
                    test_data.append(row)

            # Copy 'derm' images
            if pd.notnull(derm_file[count]):
                src2 = os.path.join(D7P_downloaded_data, 'release_v0/images', derm_file[count])
                dst2 = os.path.join(outputfolder, derm_file[count][4:])
                if os.path.isfile(src2):
                    shutil.copyfile(src2, dst2)

                row = [image_name_derm, f"{folder_name}/{image_name_derm}"] + list(one_hot.values())
                if count in train_indices:
                    train_data.append(row)
                elif count in valid_indices:
                    valid_data.append(row)
                elif count in test_indices:
                    test_data.append(row)
            

# Convert data into DataFrames
columns = ['image_name', 'Path'] + list(diagnosis_to_folder.values())
train_df = pd.DataFrame(train_data, columns=columns)
valid_df = pd.DataFrame(valid_data, columns=columns)
test_df = pd.DataFrame(test_data, columns=columns)

# Save to CSV files
train_csv_path = os.path.join(D7P_save_dir, 'train.csv')
valid_csv_path = os.path.join(D7P_save_dir, 'valid.csv')
test_csv_path = os.path.join(D7P_save_dir, 'test.csv')

train_df.to_csv(train_csv_path, index=False)
valid_df.to_csv(valid_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)