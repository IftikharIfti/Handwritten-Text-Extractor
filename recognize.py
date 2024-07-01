import os
import re
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from tkinter import Tk, Label, Button, messagebox

DETECT_FOLDER = 'test_images2'

# Load model and necessary components
np.random.seed(42)
tf.random.set_seed(42)

with open("./characters", "rb") as fp:
    characters = pickle.load(fp)

AUTOTUNE = tf.data.AUTOTUNE
char_to_num = StringLookup(vocabulary=characters, mask_token=None)
num_to_chars = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Model parameters
image_width = 128
image_height = 32
max_len = 21

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    pad_height_top = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top
    pad_width_left = pad_width // 2
    pad_width_right = pad_width - pad_width_left

    image = tf.pad(image, paddings=[[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def process_images_2(image_path):
    image = preprocess_image(image_path)
    return {"image": image}

def prepare_test_images(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths)).map(process_images_2, num_parallel_calls=AUTOTUNE)
    return dataset.batch(1).cache().prefetch(AUTOTUNE)

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

custom_objects = {"CTCLayer": CTCLayer}
reconstructed_model = keras.models.load_model("./ocr_model_100_epoch.h5", custom_objects=custom_objects)
prediction_model = keras.models.Model(reconstructed_model.get_layer(name="image").input, reconstructed_model.get_layer(name="dense2").output)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_chars(res)).numpy().decode("utf-8")
        output_text.append(res)

    return output_text

def recognize_text():
    detected_images = [os.path.join(DETECT_FOLDER, img) for img in os.listdir(DETECT_FOLDER) if os.path.isfile(os.path.join(DETECT_FOLDER, img))]
    detected_images.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    # Prepare and predict the detected images
    inf_images = prepare_test_images(detected_images)
    pred_test_text = []

    for batch in inf_images:
        batch_images = batch["image"]
        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        pred_test_text.extend(pred_texts)

    sentence = ' '.join(pred_test_text)
    return sentence

def show_recognized_text():
    sentence = recognize_text()
    messagebox.showinfo("Recognized Text", sentence)

def main():
    root = Tk()
    root.title("Handwriting Recognition")
    root.geometry("400x200")

    label = Label(root, text="Handwriting Recognition", font=("Arial", 16))
    label.pack(pady=20)

    recognize_button = Button(root, text="Recognize Text", command=show_recognized_text)
    recognize_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
