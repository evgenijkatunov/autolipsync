from subprocess import call
import os


def convert(path):
    files = os.listdir(path)
    print('--------------')
    for file in files:
        print(file)
        filepath = path+'/'+file
        if os.path.isdir(filepath):
            convert(filepath)
        if os.path.isfile(filepath):
            filename, file_extension = os.path.splitext(filepath)
            if file_extension == '.WAV':
                call(['/home/evgenij/sph2pipe/sph2pipe', filepath, '-f', 'wav', filename+'.wav'])
convert('/home/evgenij/timit/raw/TIMIT')