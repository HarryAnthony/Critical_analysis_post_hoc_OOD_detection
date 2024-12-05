# BreastMNIST
BreastMNIST [4,5] is a Breast Ultrasound Dataset from the collection [MedMNIST](https://medmnist.com/). The BreastMNIST dataset has three classes: Normal (no lesion), benign (bengin tumour), malignant (malignant tumour). The BreastMNIST dataset contains 780 images. BreastMNIST data can be downloaded [here](https://zenodo.org/records/10519652) or using the MedMNIST dataset API: 
```
pip install medmnist
```
Once downloaded, follow these steps:
- a) Extract the files from `breastmnist_224.npz` and place them into the directory `data/breastmnist`. This should include .npy files of images and labels for test, train and val.
- b) Go to the directory `data/breastmnist`, and run the python3 file `process_breastmnist.py`. This will create csv files train, test and valid which are compatable with the repository's code.
The code will classify the images into three classes (normal, benign, malignant) instead of the two classes from the breastmnist dataset, see [here for details](https://medmnist.com/). These labels were derived from [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data). If there are any issues, feel free to reach out!