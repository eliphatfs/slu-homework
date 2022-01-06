import sys, os, time, gc
from torch.optim import Adam
import json

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from utils.projection import OntologyProjection
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
projector = OntologyProjection(args.dataroot)
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path, 1)
dev_dataset = Example.load_dataset(dev_path, 0)
test_dataset = Example.load_dataset(test_path, 0)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
args.projector = projector


model = SLUTagging(args).to(device)
model.load_state_dict(torch.load("model.bin", map_location=device)['model'])
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev', 'test']
    model.eval()
    choice_id = ['train', 'dev', 'test'].index(choice)
    args.batch_size = 1
    dataset = [train_dataset, dev_dataset, test_dataset][choice_id]
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=choice != 'test')
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            if label is not None:
                labels.extend(label)
            if loss is not None:
                total_loss += loss
            count += 1
        if len(labels):
            metrics = Example.evaluator.acc(predictions, labels)
        else:
            metrics = None
    torch.cuda.empty_cache()
    gc.collect()
    return predictions, metrics, total_loss / count


start_time = time.time()
predictions, metrics, dev_loss = decode('test')
if metrics is not None:
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))


with open(os.path.join(args.dataroot, 'test_unlabelled.json'), 'r', encoding='utf-8') as fi:
    test_set = json.load(fi)
    points = [point for x in test_set for point in x]
for point, pred in zip(points, predictions):
    point['pred'] = [p.split('-') for p in pred]

with open(os.path.join(args.dataroot, 'test.json'), 'w', encoding='utf-8') as fo:
    json.dump(test_set, fo, indent=4, ensure_ascii=False)
