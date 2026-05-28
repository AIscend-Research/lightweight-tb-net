import os, glob, cv2
from preprocessing import preprocess_image

datapath = r'C:\Users\Jonathan Liu\Downloads\archive\TB_Chest_Radiography_Database'
outputpath = 'data'

filenames = []
for f in glob.glob1(os.path.join(datapath, 'Normal'), '*.png'):
    filenames.append(os.path.join(datapath, 'Normal', f))
for f in glob.glob1(os.path.join(datapath, 'Tuberculosis'), '*.png'):
    filenames.append(os.path.join(datapath, 'Tuberculosis', f))

print(f'Found {len(filenames)} images')
count = 0
for filename in filenames:
    image = preprocess_image(filename)
    savepath = os.path.join(outputpath, os.path.basename(filename))
    cv2.imwrite(savepath, image)
    if count % 500 == 0:
        print(f'{len(filenames) - count} images remaining')
    count += 1
print('Done!')