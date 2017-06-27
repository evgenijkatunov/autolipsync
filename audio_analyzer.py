import wave
import numpy.fft as fft
import struct
from Entities.Frame import Frame
import phoneme_map
import os

class Wave:
    _frame_duration = 0.05
    framerate = 0
    wave = []
    frames = []

    def __init__(self):
        self._frame_duration = 0.1
        self.framerate = 0
        self.wave = []
        self.frames = []
        self.teach_mode = False

    def load(self, path):
        wav = wave.open(path, 'rb')
        phoneme_data = None
        if self.teach_mode:
            path_to_data = path.rsplit('.', 1)[0] + '.PHN'
            print(path_to_data)
            phoneme_data = open(path_to_data, 'r')
            self.teach_data = []
            for line in phoneme_data:
                datas = line.rstrip().split(' ')
                self.teach_data.append([int(datas[0]), int(datas[1]), datas[2]])
            #for lint in self.teach_data:
            #    print(lint)
        self.framerate = wav.getframerate()
        frames_in_window = int(self.framerate * self._frame_duration)
        step_frames = int(frames_in_window / 2)
        total_frames = wav.getnframes()
        frames = wav.readframes(total_frames)
        float_frames = struct.unpack("%ih" % (total_frames * wav.getnchannels()), frames)
        float_frames = [float(val) / pow(2, 15) for val in float_frames]
        #print(float_frames)
        current_frame = 0
        # float_frames = float_frames[::2]
        while current_frame < total_frames:
            end_frame = current_frame + self._frame_duration
            count = current_frame + frames_in_window
            if count >= total_frames:
                count = total_frames - 1
            phonemes = [0 for i in range(0, len(phoneme_map.phoneme_code))]
            if self.teach_mode:
                for teach_data in self.teach_data:
                    phoneme = teach_data[2]
                    phoneme_in_frame_len = 0
                    if current_frame <= teach_data[0] <= end_frame and teach_data[1] <= end_frame:
                        phoneme_in_frame_len = float(teach_data[1] - teach_data[0])/self._frame_duration
                    elif current_frame <= teach_data[0] <= end_frame <= teach_data[1]:
                        phoneme_in_frame_len = float(end_frame - teach_data[0]) / self._frame_duration
                    elif teach_data[0] <= current_frame <= teach_data[1] <= end_frame:
                        phoneme_in_frame_len = float(teach_data[1] - current_frame) / self._frame_duration
                    elif teach_data[0] <= current_frame <= end_frame <= teach_data[1]:
                        phonemes[phoneme_map.phoneme_code[phoneme]] = 1
                    if phoneme_in_frame_len > 0.5:
                        phonemes[phoneme_map.phoneme_code[phoneme]] = 1

            new_frame = Frame(float_frames[current_frame:count], current_frame / self.framerate,
                                    self._frame_duration, self.framerate, phonemes)
            self.frames.append(new_frame)
            # if self.teach_mode:

            current_frame += step_frames
        wav.close()

    def analize(self):
        for frame in self.frames:
            frame.fft()
            frame.mel()

    def get_phoneme_map(self):
        cur_frame = {'start_time': self.frames[0].start_time,
                     'phoneme': phoneme_map.get_cmu_phoneme(self.frames[0].phoneme)}
        phonemes = [cur_frame]
        for frame in self.frames:
            frame_phoneme = phoneme_map.get_cmu_phoneme(frame.phoneme)
            if frame_phoneme != cur_frame['phoneme']:
                cur_frame = {'start_time': frame.start_time,
                             'phoneme': frame_phoneme}
                phonemes.append(cur_frame)
        return phonemes
