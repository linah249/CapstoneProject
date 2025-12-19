# models.py
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def build_mlp(input_dim, num_classes, weight_decay=1e-4):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn1d(channels=10, samples=512, num_classes=8):
    inp = Input(shape=(channels, samples))
    x = Conv1D(filters=64, kernel_size=9, strides=1, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
