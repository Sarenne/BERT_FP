import time
import argparse
import pickle
import os
# from BERT_finetuning import NeuralNetwork
from setproctitle import setproctitle

setproctitle('BERT_FP')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#Dataset path.
FT_data={
    'switchboard_nxt': 'spoken_data/switchboard_nxt_dataset_samples.pkl',
    'switchboard_ttd_nxt': 'spoken_data/switchboard_ttd_nxt_dataset_samples.pkl',
    'switchboard' : 'spoken_data/switchboard_dataset_samples.pkl',
    'ubuntu': 'ubuntu_data/ubuntu_dataset_1M.pkl',
    'douban': 'douban_data/douban_dataset_1M.pkl',
    'e_commerce': 'e_commerce_data/e_commerce_dataset_1M.pkl'
}
print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--true_batch_size",
                    default=None,
                    type=int,
                    help="The batch size to accumulate gradients by. Default is to not accumulate gradients.")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="The batch size to compute loss by.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adamw.")
parser.add_argument("--epochs", 
                    default=2,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    # default="./Fine-Tuning/FT_checkpoint/",
                    default="./fine-tune/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--checkpoint_path",
                    default="",
                    type=str,
                    help="The path to load model from checkpoint.")
parser.add_argument("--score_file_path",
                    default="./Fine-Tuning/scorefile.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--concat", 
                   default=False,
                   type=bool,
                   help="Run concat version of BERT-FP (modified) or False for original"
                  )
parser.add_argument("--joint",
                   default=False,
                   type=bool,
                   help="Run joint version of BERT-FP (modified) or False for original"
                  )
parser.add_argument("--linear_encoder",
                   default=True,
                   type=bool,
                   help="Should the classification encoder be linear or non-linear?"
                  )

args = parser.parse_args()

# If save_path doesn't exist, then create it.
CHECK_FOLDER = os.path.isdir(args.save_path)
if not CHECK_FOLDER:
    os.makedirs(args.save_path)
    print('Made save_path dir')

args.save_path += args.task + '.' + "0.pt"
args.score_file_path = args.score_file_path

print(args)
print("Task: ", args.task)

# Load the original or concat version of BERT-FP
if args.concat:
    from BERT_concat_finetuning import NeuralNetwork
elif args.joint:
    from BERT_joint_finetuning import NeuralNetwork
else:
    from BERT_finetuning import NeuralNetwork

def train_model(train, dev):
    print('TRAINING MODEL')
    model = NeuralNetwork(args=args)

    import IPython
    IPython.embed()

    model.fit(train, dev)


def test_model(test):
    print('TESTING MODEL')
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)   
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    start = time.time()

    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    if args.is_training==True:
        train_model(train,dev)
        test_model(dev)
        test_model(test)
    else:
        test_model(dev)
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




