import random
import matplotlib.pyplot as plt

def get_graph(file: str): 
    graph = {}
    with open(file, 'r') as file:
        for line in file:
            nodes = line.strip().split()
            node1, node2 = int(nodes[0]), int(nodes[1])
            if node1 not in graph:
                graph[node1] = set([node2])
            else:
                graph[node1].add(node2)
            if node2 not in graph:
                graph[node2] = set([node1])
            else:
                graph[node2].add(node1)
    return graph


def generate_valuations(graph: dict, item_count: int, auction_type):
    valuations = {}
    match auction_type:
        case "hom_single":
            for i in graph:
                random_float= random.gauss(500,150)
                valuations[i] = max(0, round(random_float))
        case "hom_mult":
            for i in graph:
                demand_count = random.randint(1, item_count)
                valuation = [random.randint(0, 1000) for _ in range(demand_count)] + [0 for _ in range(item_count-demand_count)]
                valuation.sort(reverse=True)
                valuations[i] = valuation
        case "heter_single":
            for i in graph:
                item_demand = [random.randint(0, 1) for _ in range(item_count)]
                demand_count = item_demand.count(1)
                random_float = random.gauss(500,150) * demand_count / item_count
                valuations[i] = (max(0, round(random_float)), item_demand)
        case 4:
            pass
    return valuations


def generate_sellers(graph: dict, seller_count: int):
    return random.sample(list(graph.keys()), seller_count)


def draw_valuations_distribution(valuations: dict, path):
    max_val = max(valuations.values())
    bin_width = 10
    bins = [i * bin_width for i in range(int(max_val / bin_width) + 2)]

    # 绘制直方图
    plt.hist(valuations.values(), bins=bins)
    plt.title('Histogram of Valuations')
    plt.xlabel('Valuation')
    plt.ylabel('Frequency')
    plt.savefig(path)
    plt.close()