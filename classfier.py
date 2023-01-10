from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Load the model
model = load_model(r'C:\Users\User\OneDrive\Desktop\Agro_API\keras_model.h5',compile=False)

# Load the labels
class_names = open(r'C:\Users\User\OneDrive\Desktop\Agro_API\labels.txt', 'r').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
#image = Image.open('DSC_0406.JPG').convert('RGB')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center

class result:
    def __init(Class : str , Confidence : str):
        Class = ""
        Confidence = ""




def classify(img : Image):
    size = (224, 224)
   
    im2 = img.resize((224,224),resample=Image.BICUBIC)

    #image = ImageOps.fit(im2, size, Image.Resampling.LANCZOS)
    rgb_im = im2.convert("RGB")
    #turn the image into a numpy array
    image_array = np.asarray(rgb_im)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print('Class:', class_name, end='')
    print('Confidence score:', confidence_score)
    r = result()
    r.Class = str(class_name)
    r.Confidence = str(confidence_score)
    return r