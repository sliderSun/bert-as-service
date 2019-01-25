def main():
    from bert_serving.server import BertServer
    from bert_serving.server.helper import get_run_args
    args = get_run_args()
    server = BertServer(args)
    server.start()
    server.join()


def benchmark():
    from bert_serving.server.benchmark import run_benchmark
    from bert_serving.server.helper import get_run_args, get_benchmark_parser
    args = get_run_args(get_benchmark_parser)
    run_benchmark(args)
