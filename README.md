Pre-trained weights for segmentation of total tumor burden (TTB) in metastatic prostate cancer imaging with PSMA PET/CT, FDG PET/CT, and LuPSMA SPECT/CT.
Models include on all cases in JNM manuscript (https://doi.org/10.2967/jnumed.125.270077) with the omission of cases in the hidden test data for DEEP-PSMA Grand Challenge (https://deep-psma.grand-challenge.org/).

To download repository with model weights, use git large file storage (lfs) from command line:
```
git lfs clone https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained.git
```

For inference with pre-trained models, use GTRC.py (command line inference) and GTRC_Infer.py (python scriptable)
Scripts 00_*.py-06_*.py are for training based on previously segmented image data and can be ignored if using pre-trained weights. 

Tested on windows with Python version 3.11 and Nvidia A4000 16GB (desktop) GPUs and 1070 8Gb (laptop) GPUs. On sample cases, processing time per scan is ~3min on desktop (A4000) and 10min laptop (1070).

Installation (Win) - 

Download source to project location (ex C:\GTRC\).
Best to Create virtual Environment for AI backend compatibility issues (pytorch, nnunet, tensorflow):
```
python -m venv [path to virtual environment folder (ex C:\GTRC\GTRC_venv)]
#Activate the environment and configure environment
C:\GTRC\GTRC_venv\Scripts\activate.bat
#update pip
python -m pip install --upgrade pip
#install pip requirements
pip install -r C:\GTRC\requirements.txt
```
This may take 15-20 minutes to complete. The inference requires a system call to the installed "nnUNetv2_predict" script/application. By default on windows this will be in the Scripts\ folder of your virtual environment location (eg C:\GTRC\GTRC_venv\Scripts\nnUNetv2_predict.exe). On linux the location will be different and may involve calling the virtual environment python binary followed by a path to the nnunet predict.py script located in the module’s lib/site-packages subfolder.) If you need to alter the location of this call, modify GTRC’s GTRC_Infer.py script to change the variable ‘nn_predict_exe’ to the appropriate location at line 26 of GTRC_Infer.py. Whatever that variable is will be the preamble to the system call. If any issues, troubleshoot by checking that the path in that variable yields an output of the nnUNet_predict helpfile when entered into the command line from your GTRC working directory.


To test overall functionality, a sample case from the DEEP-PSMA Dataset is available at https://zenodo.org/records/18150034
https://zenodo.org/records/18150034/files/train_0001.zip?download=1

This will contain one PSMA and FDG scan in nifti format for testing inference functionality. Extract to a folder "sample_data\train_001\"

Command to test installation on DEEP-PSMA sample case once downloaded
```
python GTRC.py --test 
```

If successful there should now be TTB niftis for each of the sample PSMA and FDG PET/CTs located in the sample_data\train_0001 folder

More specific example command:
```
python GTRC.py --ct sample_data\train_0001\PSMA\CT.nii.gz --pet sample_data\train_0001\PSMA\PET.nii.gz --tracer psma_pet --output psma_test.nii.gz --suv_threshold 3.5 -f 0
```
will run inference on the sample PSMA PET using a threshold of 3.5 and fold 0 of the nnU-net and classification models.

For scripting purposes, the inference command can be called from the function in GTRC_Infer.py file ("run_gtrc_infer(...)"). 
