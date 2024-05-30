# Artificial intelligence-enabled ECG for left ventricular diastolic function and filling pressure

This repository provides the official PyTorch implementation of the following paper: [Artificial intelligence-enabled ECG for left ventricular diastolic function and filling pressure](https://www.nature.com/articles/s41746-023-00993-7)

---

## Diastolic function grading (ground truth; label)
To label diastolic function grades, we followed a revised unified algorithm for assessment of diastolic filling pressure and function in [the paper](https://doi.org/10.1038/s41746-023-00993-7). Normal and grade 1 diastolic function were considered as normal filling pressure, and grade 2 and grade 3 diastolic functions were considered as increased filling pressure.

## AI-ECG model
ResNet18 was trained with a learning rate of 0.001 and adam optimizer for 20 epochs. 
The validation performance was converged before 20th epoch. 
The final model was chosen according to the AUC value from the validation set for increased filling pressure. 
The model was trained as a multi-class model with four outputs representing the four grades of diastolic function ('label' in the provided csv file) and the sum of four outputs was 1. Normal and grade 1 were considered normal filling pressure, and grades 2 and 3 were considered increased filling pressure. While the model outputs four values, the sum of the outputs of normal and grade 1 represents the output of normal filling pressure and the sum of grades 2 and 3 outputs represents the output of increased filling pressure. Likewise, the sum of the outputs of normal and increased filling pressures was 1. Using the sum of each two classes, we converted the multi-class model to a binary model and we applied the Youden index for the final output value.  

## Requirements for running 'main.py'
* Development conda environments
  * Please find requirements.txt.

* Files
  * Numpy ECG file with a shape of (# of ECGs, 5000, 12, 1).
  * Numpy label file with a shape of (# of ECGs,).
  * Csv file for numpy order having 'split' column (training/validation/test).

## Commands to run
```
### Training
# 12-lead
python main.py --arch resnet18 --ep 30 --lr 0.001 --ecg rhythm --mode training --data_path [NUMPY ECG FILE PATH] --label_path [NUMPY LABEL FILE PATH] --split_path [CSV PATH FOR NUMPY ORDER AND META INFORMATION]

# Single lead (e.g., lead I)
python main.py --arch resnet18 --ep 30 --lr 0.001 --ecg rhythm --num_leads 1 --specific_lead 1 --mode training --data_path [NUMPY ECG FILE PATH] --label_path [NUMPY LABEL FILE PATH] --split_path [CSV PATH FOR NUMPY ORDER AND META INFORMATION]

# 12-lead median
python main.py --arch resnet18 --ep 30 --lr 0.001 --ecg median --mode training --data_path [NUMPY ECG FILE PATH] --label_path [NUMPY LABEL FILE PATH] --split_path [CSV PATH FOR NUMPY ORDER AND META INFORMATION]

# Single lead median 
python main.py --arch resnet18 --ep 30 --lr 0.001 --ecg median --num_leads 1 --specific_lead 1 --mode training --data_path [NUMPY ECG FILE PATH] --label_path [NUMPY LABEL FILE PATH] --split_path [CSV PATH FOR NUMPY ORDER AND META INFORMATION]

### Validation command example for 12-lead ECG
python main.py --arch resnet18 --ep 30 --lr 0.001 --mode validation --saved_weight_dir ./results/[MODEL DIR NAME]
python main.py --arch resnet18 --ep 30 --lr 0.001 --mode test --saved_weight_dir ./results/[MODEL DIR NAME]
python main.py --arch resnet18 --ep 30 --lr 0.001 --mode all --saved_weight_dir ./results/[MODEL DIR NAME]
```
