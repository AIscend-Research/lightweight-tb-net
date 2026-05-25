import os, glob, cv2
from preprocessing import preprocess_image

datapath = r'C:\Users\Jonathan Liu\Downloads\archive\TB_Chest_Radiography_Database'
outputpath = 'data'

filenames = glob.glob(os.path.join(datapath, 'Tuberculosis', '*.png'))
print(f'Found {len(filenames)} TB images')

count = 0
for filename in filenames:
    image = preprocess_image(filename)
    # Rename from Tuberculosis-1.png to TB-1.png
    basename = os.path.basename(filename).replace('Tuberculosis-', 'TB-')
    savepath = os.path.join(outputpath, basename)
    cv2.imwrite(savepath, image)
    if count % 100 == 0:
        print(f'{len(filenames) - count} images remaining')
    count += 1

print('Done!')