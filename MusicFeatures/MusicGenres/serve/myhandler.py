import torch
from ts.torch_handler.base_handler import BaseHandler
import base64
import io
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
import torchvision
from pydub import AudioSegment

class myhandler(BaseHandler):
    def __init__(self):
        super(myhandler, self).__init__()
        self.sample_rate = 44100

    def initialize(self, context):
        super().initialize(context)
        self._context = context
        self.initialized = True

    def preprocess(self, data):
        print(type(data))
        for row in data:
            print(row.keys())
            audio = row.get("data") or row.get("body")
            print("----------------->raw audio data:", type(audio))
            if isinstance(audio, str):
                # if the image is a string of bytesarray.
                audio = base64.b64decode(audio)
                print("----------------->base64 decoded audio data:", type(audio))
            else:
                print("----------------->data not str")
            # If the image is sent as bytesarray
            if isinstance(audio, (bytearray, bytes)):
                if audio[:3].decode('utf-8') == 'ID3':
                    AudioSegment(audio['data'], sample_width=2, frame_rate=44100, channels=2)
                else:
                    audio = io.BytesIO(audio)
                    audio = sf.read(audio)
            else:
                # if the image is a list
                audio = torch.FloatTensor(audio)

            data = torch.sum(audio, axis=0).unsqueeze(0)
            MFCC_tranfromer = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, log_mels=True,
                                                         melkwargs={'n_fft': 1200, 'win_length': 1200,
                                                                    'normalized': True})
            MFCC_Norm = torchvision.transforms.Normalize((0.5,), (0.5,))
            data = torch.nn.functional.pad(data, (
            0, data.shape[1] // (self.sample_rate * 5) * self.sample_rate * 5 + 24000 - data.shape[1]))
            mfccs = MFCC_tranfromer(data)
            mfccs = MFCC_Norm(mfccs.unsqueeze(0))
            print("-----------------------------------> proprocess", mfccs.shape)
            return mfccs

    def postprocess(self, data):
        return data

    def inference(self, data):
        self.model.eval()
        self.model.cuda()
        predict_list = []
        with torch.no_grad():
            sum_y_ = torch.zeros(1, 8)
            for i in range(data.shape[1] // (self.sample_rate * 5)):
                test_data = data[:, :, :, i * 368:(i + 1) * 368]
                # test_data = next(iter(train_loader))[0][0]
                print("-----------------------------------> inference", data.shape)
                y_ = self.model(test_data.cuda())
                predict_list.append((torch.nn.functional.sigmoid(y_) > 0.5).view(-1).nonzero().tolist())
                sum_y_ += y_.cpu()
            sum_y_ /= (data.shape[1] // (self.sample_rate * 5) - 1)
            overall_genre = (torch.nn.functional.sigmoid(sum_y_) > 0.5).view(-1).nonzero().tolist()

        return [sum_y_, overall_genre]

    def handle(self, data, context):
        processed_data = self.preprocess(data)
        res = self.inference(processed_data)
        return self.postprocess(res)
