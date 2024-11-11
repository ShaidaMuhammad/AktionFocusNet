import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, GRU, BatchNormalization, Bidirectional, \
    GlobalAveragePooling1D, Multiply, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


def load_trained_model(model_path):
    """Loads and returns the trained model from the specified path."""
    model = load_model(model_path)
    return model

def attention_module(input_layer, hidden_units):
    attention = Dense(hidden_units, activation='tanh')(input_layer)
    attention = Dense(1, activation='softmax')(attention)
    return Multiply()([input_layer, attention])


def multi_head_attention_module(input_layer, num_heads, head_hidden_units):
    heads = [attention_module(input_layer, head_hidden_units) for _ in range(num_heads)]
    return Concatenate(axis=-1)(heads)


def create_model(input_shape, num_classes, num_attention_heads=3, head_hidden_units=128):
    signal_input = Input(shape=input_shape, name='data')
    gru1 = Bidirectional(GRU(64, return_sequences=True))(signal_input)
    bn = BatchNormalization()(gru1)

    bn = multi_head_attention_module(bn, num_heads=num_attention_heads, head_hidden_units=head_hidden_units)

    layer_1_a = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')(bn)
    layer_1_a = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(layer_1_a)

    layer_2_a = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')(bn)
    layer_2_a = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(layer_2_a)

    mid_1_a = Concatenate(axis=2)([layer_1_a, layer_2_a])
    drop = Dropout(0.5)(mid_1_a)

    before_flat = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')(drop)
    globel_average = GlobalAveragePooling1D()(before_flat)
    output = Dense(num_classes, activation='softmax')(globel_average)

    model = Model(inputs=signal_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
