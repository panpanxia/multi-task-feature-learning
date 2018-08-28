import datasets
import mtfl_train

parser = mtfl_train.parser
FLAGS = parser.parse_args()
dataset = 'MTFL'

eben = datasets.route(dataset)
print("naber")