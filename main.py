import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from PIL import Image
import io

app = FastAPI()

# تعریف مدل FCN
def create_fcn(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))
    return model

# بارگذاری و پیش‌پردازش مجموعه داده CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# تعریف ورودی مدل
input_shape = x_train.shape[1:]

# ایجاد و آموزش مدل FCN
model = create_fcn(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array)
    segmented_image = (prediction[0] > 0.5).astype(np.uint8) * 255
    
    segmented_image_pil = Image.fromarray(segmented_image.squeeze(), mode='L')
    buf = io.BytesIO()
    segmented_image_pil.save(buf, format='PNG')
    byte_im = buf.getvalue()
    
    return JSONResponse(content={"segmented_image": byte_im.hex()})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
