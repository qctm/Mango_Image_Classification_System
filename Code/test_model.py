# 1.13 Chay ket qua
from numpy import argmax
from tensorflow.keras.preprocessing import image
from keras.models import load_model

model = load_model('duong dan model')

img_path = "duong dan anh"

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)

pred = model.predict(img_preprocessed)
Res = argmax(pred, axis=1)
print(pred)

Result_Text = "{0}({1})".format(categories[Res[0]], round(pred[0][Res[0]]*100,2))

plt.text(10,10, Result_Text, color= "blue", fontsize="large", bbox=dict(fill = False, edgecolor='red', linewidth=1))
plt.imshow(img)
plt.show()
print(categories[Res[0]], pred[0][Res[0]]*100)
