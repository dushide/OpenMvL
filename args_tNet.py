
import argparse
import sys


## Parameter setting
def parameter_parser():
    parser = argparse.ArgumentParser()

    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)

    parser.add_argument("--data_path", type=str, default="../data/", help="Path of datasets.")


    parser.add_argument("--save_path", type=str, default="./result/unseen=1.txt", help="Save experimental result.")
    parser.add_argument("--save_resp_path", type=str, default="./resp/", help="Save experimental result.")
    parser.add_argument("--config_name", type=str, default="unseen1_l1.txt", help="Save experimental result.")
    parser.add_argument("--save_prob", action='store_true', default=True, help="Save experimental result.")
    parser.add_argument("--save_results", action='store_true', default=True, help="Save experimental result.")
    parser.add_argument("--save_total_results", action='store_true', default=False, help="Save experimental result.")
    parser.add_argument("--use_hypergraph", action='store_true', default=True, help="Save experimental result.")


    parser.add_argument("--unseen_num", type=int, default=1)
    parser.add_argument("--unseen_label_index", type=int, default=-100)
    parser.add_argument("--fusion_type", type=str, default="trust", help="Fusion Methods: trust average weight attention")
    parser.add_argument("--active", type=str, default="l1", help="l21 or l1")
    # the type of regularizer with Prox_h()

    parser.add_argument("--device", default="cpu", type=str, required=False)
    parser.add_argument("--fix_seed", action='store_true', default=True, help="")
    parser.add_argument("--use_softmax", action='store_true', default=True, help="")
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 42.")
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument("--ratio", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--training_rate", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--valid_rate", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--weight_decay", type=float, default=0.15, help="Weight decay")

    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--block', type=int, default=1, help='block') # for the example dataset, block can set 2 and more than 2
    parser.add_argument('--thre', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--alpha_1', type=float, default=10000)
    parser.add_argument('--beta_1', type=float, default=1)
    return parser.parse_args()