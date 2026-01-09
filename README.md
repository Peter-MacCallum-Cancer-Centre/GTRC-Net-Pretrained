Pre-trained weights for segmentation of total tumor burden (TTB) in metastatic prostate cancer imaging with PSMA PET/CT, FDG PET/CT, and LuPSMA SPECT/CT
Models trained on all cases in JNM manuscript but retrained to omit cases including in hidden test data for DEEP-PSMA Grand Challenge. Overall performance should be very similar to manuscript report.

Scripts 00_*.py-06_*.py are for training based on segmented image data. For inference with pre-trained models, use GTRC.py (command line inference) and GTRC_Infer.py (python scriptable)

Tested with Python version 3.11

Installation (Win)
Download source to project location (ex C:\GTRC\)
Best to Create virtual Environment for AI backend compatibility issues (pytorch, nnunet, tensorflow)
...
>python -m venv [path to virtual environment folder (ex C:\GTRC\GTRC_venv)]
#Activate the environment and configure environment
>C:\GTRC\GTRC_venv\Scripts\activate.bat
#update pip
>python -m pip install --upgrade pip
#install pip requirements
>pip install -r C:\GTRC\requirements.txt
#this may take 15-20 minutes to complete. The inference requires a system call to the installed nnunet_predict script/application. By default on windows this will be in the Scripts\ folder of your virtual environment location (eg C:\GTRC\GTRC_venv\Scripts\nnUNetv2_predict.exe). On linux the location will be different and may involve calling the virtual environment python application followed by a path to the nnunet predict.py script located in the module’s lib/site-packages subfolder.) If you need to alter the location of this call, modify GTRC’s GTRC_Infer.py script to change the variable ‘nn_predict_exe’ to the appropriate location. Whatever that variable is will be the preamble to the system call so test that it yields an output of the nnUNet_predict helpfile before proceeding.
'''

To test functionality, a sample case from the DEEP-PSMA Dataset is available at https://zenodo.org/records/18150034
https://zenodo.org/records/18150034/files/train_0001.zip?download=1

This will contain one PSMA and FDG scan in nifti format for testing inference functionality. 

Command to test installation on DEEP-PSMA sample case once downloaded
>python GTRC.py --test 

More specific example command:
python GTRC.py --ct sample_data\train_0001\PSMA\CT.nii.gz --pet sample_data\train_0001\PSMA\PET.nii.gz --tracer psma_pet --output psma_test.nii.gz --suv_threshold 3.5 -f 0
will run inference on the sample PSMA PET using a threshold of 3.5 and fold 0 of the nnU-net and classification models.

For scripting purposes, the inference command can be called from the function in GTRC_Infer.py file. 
