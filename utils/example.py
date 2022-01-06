import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path, word2vec=cls.word2vec.word2vec)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']
        self.mtr = ex['manual_transcript']\
            .replace("(unknown)", "")\
            .replace("(robot)", "")\
            .replace("(dialect)", "")\
            .replace("(noise)", "")\
            .replace("(side)", "")[:len(self.utt)]
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1 and bidx < len(self.utt):
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        self.denoi_idx = [Example.word_vocab[c] for c in self.mtr]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
