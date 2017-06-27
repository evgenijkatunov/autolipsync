from lstm import PhonemeRecognition
from audio_analyzer import Wave
import json
from flask import Flask, request, abort, json
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = 'waves'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/phonemes', methods=['POST'])
def post_phonemes():
    print(request.files)
    if 'wave' not in request.files:
        abort(400)
    file = request.files['wave']
    tf = tempfile.NamedTemporaryFile(dir=UPLOAD_FOLDER)
    tmp_filename = tf.name
    tf.close()
    file.save(tmp_filename)
    wave = Wave()
    wave.load(tmp_filename)
    recogn = PhonemeRecognition()
    recogn.load('model_en_full')
    recogn.predict(wave)
    result = {'phonemes': wave.get_phoneme_map()}
    print(result)
    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')