from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras import metrics
import keras.layers.merge as merge
import audio_analyzer
import phoneme_map
import audio_analyzer
import os
import numpy as np


class PhonemeRecognition:
    _features_count = 15
    _phonemes_count = 61
    _batch_size = 32
    _max_frames = 200

    def _dir_process(self, path):
        files = os.listdir(path)
        for file in files:
            filepath = path + '/' + file
            if os.path.isdir(filepath):
                self._dir_process(filepath)
            if os.path.isfile(filepath):
                filename, file_extension = os.path.splitext(filepath)
                if file_extension == '.wav':
                    print(filepath)
                    a = audio_analyzer.Wave()
                    a.load(filepath)
                    a.analize()
                    xfeatures = []
                    yfeatures = []
                    for frame in a.frames:
                        xfeatures.append(frame.mfcc)
                        yfeatures.append(frame.phonemes)
                    while len(xfeatures) < self._max_frames:
                        xfeatures.append([0 for i in range(0, self._features_count)])
                        yfeatures.append([1 if i == 60 else 0 for i in range(0, len(phoneme_map.phoneme_code))])
                    self._x.append(xfeatures)
                    self._y.append(yfeatures)

    def init(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(126, return_sequences=True), 'sum',
                                     input_shape=(self._max_frames, self._features_count)))
        self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(units=self._phonemes_count, activation='softmax')))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=[metrics.categorical_accuracy])

    def load(self, path):
        self.model = load_model(path)

    def save(self, path):
        self.model.save(path)

    def fit(self, path_fit_data):
        self._x = []
        self._y = []
        self._dir_process(path_fit_data)
        self.model.fit(
            np.asarray(self._x), np.asarray(self._y),
            batch_size=self._batch_size,
            epochs=20
        )

    def _predict_to_phonemes(self, predicts):
        phonemes = []
        for predict in predicts[0]:
            max_probability = (0, '')
            i = 0
            for case in predict:
                if case > max_probability[0]:
                    max_probability = (case, phoneme_map.code_phoneme[i])
                i += 1
            phonemes.append(max_probability[1])
        return phonemes

    def predict(self, wave):
        wave.analize()
        x = []
        for frame in wave.frames:
            x.append(frame.mfcc)
        while len(x) < self._max_frames:
            x.append([0 for i in range(0, self._features_count)])
        phonemes = self._predict_to_phonemes(self.model.predict(np.asarray([x])))
        for i in range(0, len(wave.frames)):
            wave.frames[i].phoneme = phonemes[i]
        return phonemes

    def test(self, path_test_data):
        self._x = []
        self._y = []
        self._dir_process(path_test_data)
        print(self.model.evaluate(
            self._x, self._y,
            batch_size=self._batch_size
        ))


def teach():
    phonemes_count = 61
    max_frames = 150
    model_path = 'model_en_full'

    a = audio_analyzer.Wave()
    # a.teach_mode = False
    a.load("/home/evgenij/timit/raw/TIMIT/TEST/DR2/FCMR0/SA1.wav")
    # a.load('36.wav')
    neural = PhonemeRecognition()
    # neural.load(model_path)
    neural.init()
    neural.fit('/home/evgenij/timit/raw/TIMIT/TRAIN/DR1')
    neural.save('model_en_full')
    # neural.test('/home/evgenij/timit/raw/TIMIT/TEST/DR2')
    results = neural.predict(a)
    y = []
    for frame in a.frames:
        y.append(frame.phonemes)
    while len(y) < max_frames:
        y.append([1 if i == 60 else 0 for i in range(0, len(phoneme_map.phoneme_code))])

    maxi = 0
    max = 0
    out = open("/home/evgenij/log.nn", 'w')
    correct_result = ''
    np.set_printoptions(threshold=np.nan)
    for result in results:
        j = 0
        # out.writelines(['-----------'])
        for frame in result:
            maxi = 0
            max = 0
            maxi_correct = 0
            max_correct = 0
            for i in range(0, phonemes_count):
                if frame[i] > max:
                    max = frame[i]
                    maxi = i
                if y[j][i] > max_correct:
                    max_correct = y[j][i]
                    maxi_correct = i
            j += 1
            out.write(phoneme_map.code_phoneme[maxi])
            correct_result = correct_result + phoneme_map.code_phoneme[maxi_correct] + ' '
            out.write(' ')
        out.write('\n')
        out.write(correct_result)
        out.write('\n')
        out.write(np.array_str(results))
    out.close()


def fast_teach():
    neural = PhonemeRecognition()
    neural.init()
    neural.fit('/home/evgenij/timit/raw/TIMIT/TRAIN')
    neural.save('model_en_full')
    neural.test('/home/evgenij/timit/raw/TIMIT/TEST')

#fast_teach()