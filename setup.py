from setuptools import setup
import sys
sys.setrecursionlimit(1500) 


APP = ['main.py']
DATA_FILES = [('.', ['best_age_detection_model_dropout_layer_with_scheduler_diff_params.pth',
                     'best_age_detection_model_dropout_layer.pth',
                     'best_age_detection_model_layer_params_googleNet.pth',
                     'best_age_detection_model_more_layers_saved.pth',
                     'best_age_detection_model.pth'])]

OPTIONS = {
    'argv_emulation': True,
    'packages': ['matplotlib', 'numpy', 'PIL', 'torch', 'torchvision', 'cv2', 'sklearn', 'seaborn', 'requests', 'chardet', 'charset_normalizer'],
    'excludes': ['PyInstaller', 'PySide2.QtSvg'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
