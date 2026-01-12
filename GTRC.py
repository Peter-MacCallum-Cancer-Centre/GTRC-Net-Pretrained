import argparse, sys, os

"""
script to run inference from command line, GTRC_Infer.py contains function to run inference which may be useful for batch scripting
To do: include dicom parser for reading in .dcm series and writing rt-structure sets
"""


parser = argparse.ArgumentParser(
                    prog='GTRC-Net PET/SPECT Tumor Burden segmentation',
                    description='Optimised tool based on nnU-net for segmentation of disesase extent on PSMA/FDG PET and LuPSMA SPECT',
                    epilog='Developed by Peter MacCallum Cancer Centre and ProsTIC Centre of Excellence')
parser.add_argument('--ct',default='',help='path to CT image file, nii/mha/etc ITK formats accepted')           # positional argument
parser.add_argument('--pet',default='',help='path to PET/SPECT image file, nii/mha/etc ITK formats accepted')      # option that takes a value
parser.add_argument('-o','--output',default='',help='path for output total tumor burden segmentation, in SITK formats (nii/mha,etc)')
parser.add_argument('-f','--fold',default='all',help='optional to designate which of folds (0-4) to use for inference, default "all"')
parser.add_argument('-t','--tracer',default='',help='Which tracer model to use "psma_pet", "fdg_pet", and "lupsma_spect" options')
parser.add_argument('--suv_threshold',type=float, default=3.0,help='Threshold to use for segmenting disease boundaries, default: 3.0 (suv>=3.0)')

parser.add_argument('--test',
                    action='store_true',help='if set will attempt PSMA & FDG PET inference on DEEP-PSMA Sample Case')  # on/off flag



args = parser.parse_args()
run_test= args.test

if not run_test:
    ct_fname=args.ct
    pt_fname=args.pet
    tracer=args.tracer
    output_fname=args.output
    fold=args.fold
    suv_threshold=float(args.suv_threshold)
    if '' in [ct_fname,pt_fname,tracer,output_fname]:
        print('Arguments --ct, --pet, --output/-o, & --tracer/-t are required unless running --test')
        print('See --help for more info')
        sys.exit()
    else:
        from GTRC_Infer import run_gtrc_infer
        ttb=run_gtrc_infer(pt_fname,ct_fname,tracer=tracer,output_fname=output_fname,
                           return_ttb_sitk=True,fold=fold,suv_threshold=suv_threshold)
                           


##run_test=args.test


if run_test:
    from GTRC_Infer import run_gtrc_infer
    print(r'Running Test Inference on DEEP-PSMA Sample Data (train_0001)')
    print(r'output segmentation files should appear in sample_data\psma_ttb.nii.gz and sample_data\fdg_ttb.nii.gz')

    ct_fname=r"sample_data/train_0001/PSMA/CT.nii.gz"
    pt_fname=r"sample_data/train_0001/PSMA/PET.nii.gz"
    suv_threshold=3.0

    if not os.path.exists(ct_fname):
        print(r'Missing sample datafiles, download and extract from https://zenodo.org/records/18150034')
        print(r'Direct link: https://zenodo.org/records/18150034/files/train_0001.zip?download=1')
        print(r'Image data should be located in sample_data\train_0001\PSMA\*.nii.gz and sample_data\train_0001\FDG\*.nii.gz')
        sys.exit()
        #sdownload_sample_data() #to be updated...

    ttb=run_gtrc_infer(pt_fname,ct_fname,tracer='psma_pet',output_fname=r"sample_data/psma_ttb.nii.gz",
                       return_ttb_sitk=True,
                       fold='all',suv_threshold=suv_threshold)

    ct_fname=r"sample_data/train_0001/FDG/CT.nii.gz"
    pt_fname=r"sample_data/train_0001/FDG/PET.nii.gz"
    suv_threshold=2.9650267106980

    ttb=run_gtrc_infer(pt_fname,ct_fname,tracer='fdg_pet',output_fname=r"sample_data/fdg_ttb.nii.gz",
                       return_ttb_sitk=True,
                       fold='all',suv_threshold=suv_threshold)
