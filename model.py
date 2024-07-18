from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D

from blocks import *

def residual_attention_unet(input_size=(256, 256, 1)):
    #通道數
    f = [16, 32, 64, 128]
    
    inputs = Input(input_size)

    #編碼塊
    encode0 = inputs
    encode1 = residual_block(encode0, f[0], strides=1)
    pooling1 = MaxPooling2D((2, 2))(encode1)
    
    encode2 = residual_block(pooling1, f[1], strides=1)
    pooling2 = MaxPooling2D((2, 2))(encode2)
    
    encode3 = residual_block(pooling2, f[2], strides=1)
    pooling3 = MaxPooling2D((2, 2))(encode3)

    #橋梁
    b0 = residual_block(pooling3, f[3], strides=1)

    #解碼塊
    up1 = upsample_concat_block(b0, encode3)
    attention1 = attention_gate(encode3, up1, f[2])
    decode1 = residual_block(attention1, f[2])

    up2 = upsample_concat_block(decode1, encode2)
    attention2 = attention_gate(encode2, up2, f[1])
    decode2 = residual_block(attention2, f[1])

    up3 = upsample_concat_block(decode2, encode1)
    attention3 = attention_gate(encode1, up3, f[0])
    decode3 = residual_block(attention3, f[0])

    #輸出
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(decode3)
    
    model = Model(inputs=inputs, outputs=outputs)

    return model