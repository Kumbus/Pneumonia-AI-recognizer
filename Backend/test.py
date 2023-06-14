import pathlib
import tensorflow as tf

model = tf.keras.models.load_model('C:/Users/Rezerwowy/Desktop/BIAI/checkpoint_10_epochs.h5')

batch_size =  32
img_size = 180
test_dir = pathlib.Path("C:/Users/Rezerwowy/Desktop/Biai/Backend/chest_xray/test")

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  image_size = (img_size, img_size),
  batch_size = batch_size,
  shuffle=False)


loss, acc, prec, rec = model.evaluate(test_ds)


