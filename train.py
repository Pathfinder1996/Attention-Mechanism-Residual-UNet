import os
import pickle

import tensorflow as tf
from keras.callbacks import *

from dataloader import *
from model import *
from mymetrics import *

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
訓練跟驗證集的圖像與標籤資料增生
可以將trainGenerator函式中save_to_dir改到自己指定資料夾, 就能看到增生的圖像與標籤
例如 save_to_dir=r'data\membrane\train\aug'
'''

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(16, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

val_aug_dict = {'rescale': 1./255}

valGene = trainGenerator(16, 'data/membrane/val', 'val_image', 'val_label', val_aug_dict,
                         image_color_mode="grayscale", mask_color_mode="grayscale",
                         image_save_prefix="val_image", mask_save_prefix="val_label",
                         save_to_dir=None, target_size=(256, 256), seed=1)

model = ResUNet_Attention()

#打印模型結構
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=dice_loss, metrics=[IOU_coefficient, dice_coefficient, 'accuracy'])

model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='loss', verbose=1, save_best_only=True)

#每個時期迭代次數 = 資料夾圖片總數 / 訓練批次(batch_size)
history = model.fit(myGene, epochs=100, steps_per_epoch=18, validation_data=valGene, validation_steps=5,
                    callbacks=[model_checkpoint])

#保存訓練指標
with open("trainHistory.txt", "wb") as file_pi:
    pickle.dump(history.history, file_pi)

#訓練指標視覺化
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(history.history['dice_coefficient'], label='Train')
axs[0, 0].plot(history.history['val_dice_coefficient'], label='Validation')
axs[0, 0].set_title('Dice Coefficient')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Dice Coefficient')
axs[0, 0].legend()

axs[0, 1].plot(history.history['loss'], label='Train')
axs[0, 1].plot(history.history['val_loss'], label='Validation')
axs[0, 1].set_title('Loss Function')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

axs[1, 0].plot(history.history['accuracy'], label='Train')
axs[1, 0].plot(history.history['val_accuracy'], label='Validation')
axs[1, 0].set_title('Accuracy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()

axs[1, 1].plot(history.history['IOU_coefficient'], label='Train')
axs[1, 1].plot(history.history['val_IOU_coefficient'], label='Validation')
axs[1, 1].set_title('IOU Coefficient')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('IOU Coefficient')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

#用訓練好的模型預測測試集圖像
testGene = testGenerator(r"D:\unet_ours\data\membrane\test")
model.load_weights("best_model.hdf5")
results = model.predict(testGene, 30, verbose=1)
saveResult(r"D:\unet_ours\data\membrane\test", results)