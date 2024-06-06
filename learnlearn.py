MAIN_EPOCH = 1
MAIN_INITIALIZE_MODEL = True
MAIN_MAX_VOCAB_SIZE = 3936
# 3936 for the first 1000 audio samples
# 51011 for the whole dataset

import librosa
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import \
    layers as keras_layers, \
    Model as keras_Model, \
    losses as keras_losses, \
    optimizers as keras_optimizers, \
    models as keras_models
    # backend as keras_backend  # keras_backend.clear-something

from fastnumpyio import unpack as fnpio_unpack

tf.config.run_functions_eagerly(False)


print('DEFINING CONSTS...')


PATH_DATASET = 'datasets/speech-commonvoice/'
PATH_MFCCS = PATH_DATASET + 'mfccs/'
PATH_BINARIES = PATH_DATASET + 'mfccs-binary-transposed/'
PATH_MP3 = PATH_DATASET + 'raw-mp3/'
### UPDATE
PATH_DATA = PATH_DATASET + 'TDATA.tsv'
PATH_VOCAB = PATH_DATASET + 'VOCAB.txt'
# PATH_DATA = PATH_DATASET + 'DATA.tsv'
# PATH_VOCAB = PATH_DATASET + 'VOCAB.txt'
PATH_TRAIN = 'train/'
PATH_MODELS = PATH_TRAIN + 'models/'
PATH_BENCH = PATH_TRAIN + 'bench/'
PATH_BENCH_TRAIN = PATH_BENCH + 'train/'
PATH_BENCH_VALID = PATH_BENCH + 'valid/'
UNITS = 128
LR = 0.001 / 4
NOF_SAMPLES = 5_000
BATCH_SIZE = 1
MAX_VOCAB_SIZE = 50_000
if MAX_VOCAB_SIZE is None:
    MAX_VOCAB_SIZE = MAIN_MAX_VOCAB_SIZE
try:
    EPOCH = int(sys.argv[1])
except:
    EPOCH = MAIN_EPOCH
try:
    INITIALIZE_MODEL = bool(int(sys.argv[2]))
except:
    INITIALIZE_MODEL = MAIN_INITIALIZE_MODEL
EPOCHS_PER_BACKUP = 1


print('INIT:', INITIALIZE_MODEL)


BENCH_SIZE = 5
# valid:
#      2 > 18849005: совет безопасности приступает к рассмотрению пункта своей повестки дня.
#   7086 > 18904763: где вы видите себя через пять лет?
#  67757 > 21649713: я должен идти.
# 124314 > 28931392: давай лучше ловить котов!
# 168733 > 37171237: плутовка к дереву на цыпочках подходит вертит хвостом, с вороны глаз не сводит
BENCH_VALID_FILES = [
    'common_voice_ru_18849005',
    'common_voice_ru_18904763',
    'common_voice_ru_21649713',
    'common_voice_ru_28931392',
    'common_voice_ru_37171237']
BENCH_VALID_TARGETS = [
    'совет безопасности приступает к рассмотрению пункта своей повестки дня .',
    'где вы видите себя через пять лет ?',
    'я должен идти .',
    'давай лучше ловить котов !',
    'плутовка к дереву на цыпочках подходит вертит хвостом , с вороны глаз не сводит']

# train:
#      1 > 18849004: слово имеет уважаемый представитель республики корея.
#  54415 > 20414689: как умная, деликатная женщина могла так унижать сестру!
#  93394 > 25241232: а это впереди, кажется, наш лес?
# 133383 > 30513396: сейчас мы вас успокоим.
# 141527 > 32182593: окно, занавешенное шторой, всё больше и больше светлело, потому что начался день.
BENCH_TRAIN_FILES = [
    'common_voice_ru_18849004',
    'common_voice_ru_20414689',
    'common_voice_ru_25241232',
    'common_voice_ru_30513396',
    'common_voice_ru_32182593']
BENCH_TRAIN_TARGETS = [
    'слово имеет уважаемый представитель республики корея .',
    'как умная , деликатная женщина могла так унижать сестру !',
    'а это впереди , кажется , наш лес ?',
    'сейчас мы вас успокоим .',
    'окно , занавешенное шторой , всё больше и больше светлело , потому что начался день .']


def interval(a, b, step=1):
    return range(a, b+1, step) if step > 0 else range(a, b-1, step)


def end(seq):
    return len(seq) - 1


def rints(mn, mx, shape=None):
    if shape is None:
        return np.random.randint(mn, mx+1)
    return np.random.randint(mn, mx+1, shape)


def ndarray_from_binary(path:str) -> np.ndarray:
    with open(path, 'rb') as inp:
        bytecode = inp.read()
    return fnpio_unpack(bytecode)


class Encoder(keras_layers.Layer):
    def __init__(self, units, **kwargs):
        super(Encoder, self).__init__()
        self.units = units
        # bidirectional rnn with memory cells
        self.rnn = keras_layers.Bidirectional(
            merge_mode='sum',
            layer=keras_layers.GRU(
                units=self.units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform'))
    
    def call(self, x):
        ''' x: file name with no extension or path '''
        x = self.rnn(x)
        return x
    
    def convert_audio(self, audio, n_mfccs=13):
        ''' audio: str or transposed mfcc array '''
        if isinstance(audio, str):
            signal, sr = librosa.load(audio)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfccs)
            dmfcc = librosa.feature.delta(mfcc)
            ddmfcc = librosa.feature.delta(mfcc, order=2)
            cmfcc = np.transpose(np.concatenate((mfcc, dmfcc, ddmfcc), axis=0))
        else:
            cmfcc = audio
        return self(tf.convert_to_tensor(cmfcc)[tf.newaxis, :, :])
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units
        }
        return {**base_config, **config}


class CrossAttention(keras_layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.units = units
        self.mha = keras_layers.MultiHeadAttention(
            num_heads=1,
            key_dim=units)
        self.layernorm = keras_layers.LayerNormalization()
        self.add = keras_layers.Add()

    def call(self, x, context):
        attention_output, attention_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)
        # caching attention_scores for later plotting
        attention_scores = tf.reduce_mean(attention_scores, axis=1)
        self.last_attention_weights = attention_scores
        x = self.add([x, attention_output])
        x = self.layernorm(x)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units
        }
        return {**base_config, **config}


class Decoder(keras_layers.Layer):
    def __init__(self, vocab_path, max_vocab_size, units, **kwargs):
        super(Decoder, self).__init__()
        self.units = units
        self.vocab_path = vocab_path
        self.max_vocab_size = max_vocab_size
        with open(vocab_path, 'r', encoding='utf-8') as inp:
            vocab = [''] + inp.readline().split()
        self.text_processor = keras_layers.TextVectorization(
            standardize=None,
            ragged=True,
            vocabulary=vocab[:max_vocab_size])
        self.vocab_size = self.text_processor.vocabulary_size()
        # text
        self.word_to_id = keras_layers.StringLookup(
            vocabulary=self.text_processor.get_vocabulary(),
            mask_token='',
            oov_token='[UNK]')
        self.id_to_word = keras_layers.StringLookup(
            vocabulary=self.text_processor.get_vocabulary(),
            mask_token='',
            oov_token='[UNK]',
            invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')
        # model
        self.embedding = keras_layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=units,
            mask_zero=True)
        self.rnn = keras_layers.GRU(
            units=self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.attention = CrossAttention(units=self.units)
        self.output_layer = keras_layers.Dense(units=self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        x = self.embedding(x)
        x, state = self.rnn(x, initial_state=state)
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        logits = self.output_layer(x)
        if return_state:
            return logits, state
        else:
            return logits
        
    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "vocab_path": self.vocab_path,
            "max_vocab_size": self.max_vocab_size
        }
        return {**base_config, **config}
        
    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(batch_size)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        ans = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return ans
    
    def get_next_token(self, context, next_token, done, state, temperature=0.0):
        logits, state = self(context, next_token, state=state, return_state=True)
        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)
        # If a sequence produces an `end_token`, set it `done`
        done = done | (next_token == self.end_token)
        # Once a sequence is done it only produces 0-padding.
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
        return next_token, done, state


class Recognizer(keras_Model):
    def __init__(self, units, vocab_path, max_vocab_size, **kwargs):
        super().__init__()
        self.units = units
        self.vocab_path = vocab_path
        self.max_vocab_size = max_vocab_size
        self.encoder = Encoder(units)
        self.decoder = Decoder(vocab_path, max_vocab_size, units)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)
        # probably not needed anymore?
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "vocab_path": self.vocab_path,
            "max_vocab_size": self.max_vocab_size
        }
        return {**base_config, **config}
    
    def recognize(self, audio, *, max_length=50, temperature=0.0, n_mfccs=13):
        ''' audio: str or transposed mfcc array '''
        context = self.encoder.convert_audio(audio=audio, n_mfccs=n_mfccs)
        # batch_size = tf.shape(texts)[0]
        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)
        for times in range(max_length):
            next_token, done, state = self.decoder.get_next_token(
                context, next_token, done, state, temperature)
            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)
            if tf.executing_eagerly() and tf.reduce_all(done):
                break
        tokens = tf.concat(tokens, axis=-1)
        self.last_attention_weights = tf.concat(attention_weights, axis=1)
        result = self.decoder.tokens_to_text(tokens)
        return result
    

class MaskedLoss(keras_losses.Loss):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funct = keras_losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None)
    
    def call(self, y_true, y_pred):
        loss = self.loss_funct(y_true, y_pred)
        mask = tf.cast(y_true != 0, loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    def get_config(self):
        base_config = super().get_config()
        config = dict()
        return {**base_config, **config}


# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN


print('LOADING DATA...')


data = pd.read_csv(PATH_DATA, sep='\t')
sample = data.sample(NOF_SAMPLES)
with open(PATH_VOCAB, 'r', encoding='utf-8') as inp:
    vocab = [''] + inp.readline().split()


print('CREATING A TEXT PROCESSOR...')


TEXT_PROCESSOR = keras_layers.TextVectorization(
    standardize=None,
    ragged=True,
    vocabulary=vocab[:MAX_VOCAB_SIZE])


print('CREATING A TRAIN DATASET...')


print('-- CONTEXT...')
context = tf.ragged.constant([
    ndarray_from_binary(PATH_BINARIES + sample.path.iloc[i])
    for i in range(NOF_SAMPLES)])
print('-- TARGET...')
target = tf.constant([sample.sentence.iloc[i]
    for i in range(NOF_SAMPLES)])
print('-- TRAIN...')
train = (
    tf.data.Dataset
    .from_tensor_slices((context, target))
    .batch(BATCH_SIZE, drop_remainder=True))

@tf.function(reduce_retracing=True)
def recompile(context, target):
    context = context.to_tensor()
    target = TEXT_PROCESSOR(target)
    target_in = target[:, :-1].to_tensor()
    target_out = target[:, 1:].to_tensor()
    return (context, target_in), target_out

train = train.map(recompile, tf.data.AUTOTUNE)


print('CREATING A MODEL...')


if INITIALIZE_MODEL:
    model = Recognizer(UNITS, PATH_VOCAB, MAX_VOCAB_SIZE)
    model.compile(
        optimizer=keras_optimizers.Adam(),
        loss=MaskedLoss())
else:
    model = keras_models.load_model(filepath=PATH_MODELS + 'model.keras')
    lr_before = model.optimizer.learning_rate.value.numpy()
    model.optimizer.learning_rate.assign(LR)
    lr_after = model.optimizer.learning_rate.value.numpy()
    print(f'LR: {lr_before} --> {lr_after}')


print('TRAINING THE MODEL...')


@tf.function(reduce_retracing=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = model.loss(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

for step, (x, y) in enumerate(train):
    loss = train_step(x, y)
    if (step+1) % 100 == 0:
        print(f'[epoch: {EPOCH:>5}] --> [step: {step+1:>5} / {NOF_SAMPLES:>5}]')


print('SAVING WEIGHTS AND TESTING...')

model.save(PATH_MODELS + 'model.keras')

if EPOCH % EPOCHS_PER_BACKUP == 0:
    model.save(PATH_MODELS + 'epoch_' + str(EPOCH) + '.keras')

# recotest = model.recognize(PATH_TEST_MP3).numpy()[0].decode()
with open(PATH_TRAIN + 'recotests.txt', 'a', encoding='utf-8') as out:
    out.write(f'\n[EPOCH {EPOCH:>3}] --> [TEST]:\n\n')
    # train
    for i in range(BENCH_SIZE):
        file_path = PATH_BENCH_TRAIN + BENCH_TRAIN_FILES[i]
        source = ndarray_from_binary(file_path)
        output = model.recognize(source).numpy()[0].decode()
        target = BENCH_TRAIN_TARGETS[i]
        out.write(f'OUTPUT: {output}\n')
        out.write(f'TARGET: {target}\n\n')
    # valid
    for i in range(BENCH_SIZE):
        file_path = PATH_BENCH_VALID + BENCH_VALID_FILES[i]
        source = ndarray_from_binary(file_path)
        output = model.recognize(source).numpy()[0].decode()
        target = BENCH_VALID_TARGETS[i]
        out.write(f'OUTPUT: {output}\n')
        out.write(f'TARGET: {target}\n\n')
    
