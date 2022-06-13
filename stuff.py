import os
from pathlib import Path

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


for t in os.listdir(os.path.join('runs','detect', get_detection_folder().split('\\')[-1], 'labels')):
    print(str(Path(f'{get_detection_folder()}') /'labels'/ t))

for img in os.listdir(get_detection_folder()):
    folder=str(Path(f'{get_detection_folder()}') / img)
    if img=='labels':
        print(folder)





