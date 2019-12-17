import torch
import librosa
from manage_audio import AudioPreprocessor
from torch.autograd import Variable
from model import SpeechResModel
import numpy as np
import os
from scipy.spatial.distance import cosine


class Test_Processor():
    def __init__(self):
        self.config = dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=19)
        self.model = SpeechResModel(self.config)
        self.model.load('../model/res15-narrow-softmax.pt')
        self.input_length = 16000
        self.audio_processor = AudioPreprocessor(n_mels=40, n_dct_filters=40, hop_ms=10)
        self.wanted_words = ['zero','one','two','three','four','five','six','seven','eight','nine']
        self.audio_path = '/users/liuli/database/speech_commands_testset'

    def compute_embedding(self, audio_file):
        """计算输入音频的embedding"""
        data = librosa.load(audio_file, sr=16000)[0]
        # pad和truncated到16000个点
        if len(data) < self.input_length:
            model_in = np.pad(data, (0, max(0, self.input_length - len(data))), "constant")
        else:
            model_in = data[len(data) - self.input_length:]
        # 计算输入网络的tensor
        input_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(model_in).reshape(1, 101, 40))
        self.model.eval()
        input_tensor = Variable(input_tensor, requires_grad=False)
        output = self.model(input_tensor)
        return self.model.embedding.detach().numpy()

    def compute_embedding_dict(self, start, end):
        """计算每个关键词的embedding字典"""
        words_list = os.listdir(self.audio_path)
        keywords_dict = {}
        for keyword in words_list:
            if keyword in self.wanted_words:
                # keyword目录
                print('computing %s' % keyword)
                keyword_path = os.path.join(self.audio_path, keyword)
                keywords_dict[keyword] = []
                # 音频目录
                utt_list = os.listdir(keyword_path)
                for i in range(start, end):
                    utt_path = os.path.join(keyword_path, utt_list[i])
                    embedding = self.compute_embedding(utt_path)
                    keywords_dict[keyword].append(embedding)
        return keywords_dict

    def compute_enrollment_dict(self, start, end):
        """计算每个关键词的注册字典"""
        enroll_dict = self.compute_embedding_dict(start, end)
        for key, value in enroll_dict.items():
            embedding = np.vstack(value)
            embedding = np.mean(embedding, axis=0).reshape([1, 19])
            enroll_dict[key] = embedding
        return enroll_dict

    @staticmethod
    def compute_accuracy(eval_dict, enroll_dict, num_sample):
        """计算模板匹配的准确率"""
        similarity_scores = {}
        true_sample = 0
        for eval_keyword, eval_reference_list in eval_dict.items():
            for eval_reference in eval_reference_list:
                for keyword, reference in enroll_dict.items():
                    cos_scores = 1 - cosine(eval_reference, reference)
                    similarity_scores.update({keyword: cos_scores})
                # 找相似度最大的key值
                if max(similarity_scores, key=similarity_scores.get) == eval_keyword:
                    true_sample += 1
        acc = true_sample / num_sample
        print('Accuray: {}'.format(acc))


if __name__ == '__main__':
    test_processer = Test_Processor()
    enroll_dict = test_processer.compute_enrollment_dict(start=0, end=10)
    eval_dict = test_processer.compute_embedding_dict(start=10, end=40)
    test_processer.compute_accuracy(eval_dict, enroll_dict, 300)

