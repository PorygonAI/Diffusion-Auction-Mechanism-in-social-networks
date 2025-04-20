import random
import numpy as np

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


# def get_valuations_from_file(file: str, item_count: int):
#     valuations = np.load(file,allow_pickle=True).item()
#     for i in valuations.keys():
#         valuation = valuations[i] + [0 for _ in range(item_count-len(valuations[i]))]
#         valuations[i] = valuation

#     return valuations


def generate_valuations(graph: dict, item_count: int, auction_type):
    valuations = {}
    match auction_type:
        case "hom_single":
            for i in graph:
                random_float= random.gauss(50,20)
                valuations[i] = max(0, round(random_float))
        case "hom_mult":
            for i in graph:
                demand_count = random.randint(1, item_count)
                valuation = [random.randint(0, 1000) for _ in range(demand_count)] + [0 for _ in range(item_count-demand_count)]
                valuation.sort(reverse=True)
                valuations[i] = valuation
        case 3:
            for i in graph:
                item_demand = [random.randint(0, 1) for _ in range(item_count)] 
                valuation = random.randint(0,1000)
                valuations[i] = (valuation, item_demand)
        case 4:
            pass
    return valuations


def generate_sellers(graph: dict, seller_count: int):
    return random.sample(list(graph.keys()), seller_count)

