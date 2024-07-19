from keras.layers import Conv2D, UpSampling2D, Concatenate, Add, BatchNormalization, Activation, Multiply

#批量歸一化與激活函數
def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    return x

#卷積塊
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

#殘差塊
def residual_block(x, filters, strides=1):
    res = conv_block(x, filters, strides=strides)
    res = conv_block(res, filters, strides=1)
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output

#注意力
def attention_gate(Fg, Fs, filters):
    Wg = Conv2D(filters, kernel_size=1, padding="same")(Fg)
    Wg = BatchNormalization()(Wg)

    Ws = Conv2D(filters, kernel_size=1, padding="same")(Fs)
    Ws = BatchNormalization()(Ws)

    psi = Activation("relu")(Add()([Wg, Ws]))
    psi = Conv2D(1, kernel_size=1, padding="same")(psi)
    psi = Activation("sigmoid")(psi)

    return Multiply()([Fs, psi])

#上採樣與跳層連接塊 編碼塊跳層連接到解碼塊
def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c