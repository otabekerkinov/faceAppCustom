# FaceAppCustom

## Description
FaceAppCustom is an application designed for age estimation from facial images. It utilizes pre-trained deep-learning models and provides a graphical user interface for easy interaction.

## Installation
To install the libraries, run this command from the root directory:

```bash
pip install -r requiremetns.txt
```

or

```bash
pip install -r requiremetns.txt
```

## Usage
Run the application with the following command from the root directory:
```bash
python main.py
```
or
```bash
python3 main.py
```
Choose the model from the dropdown menu, then choose an image, video file, or start a live video to detect faces and predict ages. You can also run the model on your preferred dataset, however, note that it should be in the proper format:
* folder containing subfolders for each age e.g folder named 001 which contains pictures of 1-year-olds, each picture has a prefix with an age, eg. 001_somehing.png. Same for other ages, example pic:

![image](https://github.com/otabekerkinov/faceAppCustom/assets/63511012/9bc5eb85-9c43-483c-aa22-2e0a9f561f95)


Example of the UI:

![image](https://github.com/otabekerkinov/faceAppCustom/assets/63511012/6c0658f4-7b82-42f3-b616-a03f1b670323)


## Note
Please keep in mind that the application and the libraries were built using Python 3.11, if you're using a different one you might encounter errors, and need to switch to the version we used.

Also if you'd like to test the training models yourself, each training has its separate file with the code, each starts with ageModel.....py e.g:
```bash
ageModelResNetMoreLayers.py
```

to start the model training just run the file from ui or terminal:
```bash
python ageModelResNetMoreLayers.py
```

it should create a file with .pth extension when training is done.

Training code uses hardcoded paths for the dataset, you might wanna change it or prepare your dataset in the same file structure we used. 

The dataset we used is not included in this repo, you can find and download it here:
https://www.kaggle.com/datasets/frabbisw/facial-age

# Release
For the Windows executable download the _internal.zip  from releases: https://github.com/otabekerkinov/faceAppCustom/releases/tag/main

You might get a popup from Windows, saying that the app is not secure to run or something similar
![windows-protected-your-pc-1-3228650663](https://github.com/otabekerkinov/faceAppCustom/assets/63511012/d40ce517-17b1-4c9f-b578-803ac31b1269)

just click on more info and a button called run anyway should appear. Alternatively, disable your antivirus or add the app in exclusions in the Windows Defender
