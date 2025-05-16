import copy
from collections import deque
import dataset
import matplotlib.pyplot as plt
import numpy as np


def bfs_from_seller(graph: dict, seller: int):
    visited = set()
    queue = deque([seller])
    result = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            result.append(node)
            visited.add(node)
            neighbors = graph[node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
    result.remove(seller)
    return result


def tarjan(graph, seller):
    index = 0
    disc = {node: -1 for node in graph}
    low = {node: -1 for node in graph}
    parent = {node: None for node in graph}
    ap = {node: False for node in graph}

    stack = [(seller, None, 0)]  # (当前节点, 父节点, 子节点计数)
    while stack:
        u, p, children = stack.pop()
        if disc[u] == -1:
            disc[u] = index
            low[u] = index
            index += 1
            parent[u] = p
            stack.append((u, p, children))  # 重新压入栈，等待处理子节点
            for v in graph[u]:
                if disc[v] == -1:
                    stack.append((v, u, 0))
                elif v != p:
                    low[u] = min(low[u], disc[v])
        else:
            # 回溯时更新 low 值
            for v in graph[u]:
                if parent[v] == u:
                    low[u] = min(low[u], low[v])
                    if parent[u] is None and children > 1:
                        ap[u] = True
                    if parent[u] is not None and low[v] >= disc[u]:
                        ap[u] = True
    return ap


def find_reachable(graph, seller, removed):
    visited = set()
    stack = [seller]
    while stack:
        node = stack.pop()
        if node not in visited and node != removed:
            visited.add(node)
            stack.extend(graph.get(node, []))
    visited.remove(seller)
    return visited


def get_top_m_buyers(graph: dict, valuations: dict, item_count: int, seller: int, removed, buyers, articulation_points, visited_buyers=None):
    if removed == None:
        buyers_valuations = {i: valuations[i] for i in buyers}
        return sorted(buyers_valuations.items(), key=lambda item: item[1], reverse=True)[:item_count]

    if (visited_buyers == None) or (removed not in visited_buyers):
        if articulation_points[removed]:
            buyers_except_i = find_reachable(graph, seller, removed)
        else:
            buyers_except_i = copy.deepcopy(buyers)
            buyers_except_i.remove(removed)
        if visited_buyers != None:
            visited_buyers[removed] = buyers_except_i
    else:
        buyers_except_i = visited_buyers[removed]

    valuations_except_i = {i :valuations[i] for i in buyers_except_i}
    top_m_buyers = sorted(valuations_except_i.items(), key=lambda item: item[1], reverse=True)[:item_count]
    return top_m_buyers


def MPA_in_all(graph: dict, valuations: dict, item_count: int, seller: int):
    buyers = find_reachable(graph, seller, None)
    buyers_valuations = {i : valuations[i] for i in buyers}
    payments = {}

    buyers_sorted = sorted(buyers_valuations.items(), key=lambda item: item[1], reverse=True)
    top_m_buyers = buyers_sorted[:item_count]
    allocation = [pair[0] for pair in top_m_buyers]
    optimal_SW = sum(pair[1] for pair in top_m_buyers)

    for i in allocation:
        payments[i] = 0 if len(buyers_sorted)<item_count+1 else buyers_sorted[item_count][1]

    seller_revenue = sum(payments.values())
    return allocation, payments, optimal_SW, seller_revenue


def MPA_in_neighbors(graph: dict, valuations: dict, item_count: int, seller: int):
    buyers = graph[seller]
    buyers_valuations = {i : valuations[i] for i in buyers}
    payments = {}

    buyers_sorted = sorted(buyers_valuations.items(), key=lambda item: item[1], reverse=True)
    top_m_buyers = buyers_sorted[:item_count]
    allocation = [pair[0] for pair in top_m_buyers]
    optimal_SW = sum(pair[1] for pair in top_m_buyers)

    for i in allocation:
        payments[i] = 0 if len(buyers_sorted)<item_count+1 else buyers_sorted[item_count][1]

    seller_revenue = sum(payments.values())
    return allocation, payments, optimal_SW, seller_revenue


def VCG(graph: dict, valuations: dict, item_count: int, seller: int):
    buyers = find_reachable(graph, seller, None) 
    articulation_points = tarjan(graph, seller)
    payments = {}

    top_m_buyers = get_top_m_buyers(graph, valuations, item_count, seller, None, buyers, articulation_points)
    allocation = [pair[0] for pair in top_m_buyers]
    optimal_SW = sum(pair[1] for pair in top_m_buyers)
    
    for i in buyers:
        top_m_buyers_except_i = get_top_m_buyers(graph, valuations, item_count, seller, i, buyers, articulation_points)
        optimal_SW_except_i = sum(pair[1] for pair in top_m_buyers_except_i)
        others_SW_except_i = optimal_SW - valuations[i] * int(i in allocation)
        payments[i] = optimal_SW_except_i - others_SW_except_i

    seller_revenue = sum(payments.values())
    return allocation, payments, optimal_SW, seller_revenue


def VCG_RM(graph: dict, valuations: dict, item_count: int, seller: int):
    buyers = find_reachable(graph, seller, None)
    payments = {}
    articulation_points = tarjan(graph, seller)
    # allocation and SW
    top_m_buyers = get_top_m_buyers(graph, valuations, item_count, seller, None, buyers, articulation_points)
    allocation = [pair[0] for pair in top_m_buyers]
    optimal_SW = sum(pair[1] for pair in top_m_buyers)
    
    for i in buyers:
        top_m_buyers_except_i = get_top_m_buyers(graph, valuations, item_count, seller, i, buyers, articulation_points)
        valuation_m_except_i = 0 if len(top_m_buyers_except_i)<item_count else top_m_buyers_except_i[-1][1]
        
        if i in allocation:
            payments[i] = valuation_m_except_i
        else:
            valuation_m = 0 if len(top_m_buyers)<item_count else top_m_buyers[-1][1]
            payments[i] = valuation_m_except_i - valuation_m

    seller_revenue = sum(payments.values())
    return allocation, payments, optimal_SW, seller_revenue


def DNA_MU_R(graph: dict, valuations: dict, item_count: int, seller: int):
    bfs_order = bfs_from_seller(graph, seller)
    buyers = bfs_order
    payments = {} 
    articulation_points = tarjan(graph, seller)
    visited_buyers = {}

    def get_allocation_i(i: int, valuation_i=None):
        new_valuations = copy.deepcopy(valuations)
        if i != None:
            new_valuations[i] = valuation_i + 0.01

        k = item_count
        allocation_i = []
        for j in bfs_order:
            if k == 0:
                break
            top_m_buyers_except_j = get_top_m_buyers(graph, new_valuations, item_count, seller, j, buyers, articulation_points, visited_buyers)
            valuation_m_except_j = 0 if len(top_m_buyers_except_j)<item_count else top_m_buyers_except_j[-1][1]
            if new_valuations[j] >= valuation_m_except_j:
                allocation_i.append(j)
                k -= 1
        return allocation_i
    
    allocation = get_allocation_i(None)
    for i in allocation:
        valuation_max = valuations[i]
        valuation_min = 0
        while(True):
            if (valuation_max == valuation_min) or \
                ((valuation_max - valuation_min == 1) and (i not in get_allocation_i(i, valuation_min))):   
                break

            valuation_mid = (valuation_max + valuation_min) // 2
            if i in get_allocation_i(i, valuation_mid):
                valuation_max = valuation_mid
            else:
                valuation_min = valuation_mid
                
        payments[i] = valuation_max
        # payments[i] = np.nan
       
    SW = sum(valuations[i] for i in allocation)
    seller_revenue = sum(payments.values())
    return allocation, payments, SW, seller_revenue


def get_priority(i: int, graph_for_priority: dict, priority_type, visited: set, seller: int):
    def get_distance(i: int, graph_for_priority: dict, seller: int):
        visited_distance = set()
        queue = deque([(seller, 0)])
        visited_distance.add(seller)
        while queue:
            current_node, distance = queue.popleft()
            if current_node == i:
                return distance                
            for neighbor in graph_for_priority[current_node]:
                if neighbor not in visited_distance:
                    queue.append((neighbor, distance + 1))
                    visited_distance.add(neighbor)

    match priority_type:
        case "constant": 
            return 0
        
        case "new_agent": 
            count = 0
            neighbors = graph_for_priority[i]
            for neighbor in neighbors:
                if (neighbor not in visited) and (neighbor != seller):
                    count += 1
            return count
        
        case "degree": 
            return len(graph_for_priority[i])
        
        case "distance":
            return get_distance(i, graph_for_priority, seller)
        
        case "negative_distance": # negative_distance
            return -get_distance(i, graph_for_priority, seller)

def MUDAN(graph: dict, valuations: dict, item_count: int, seller: int, graph_for_priority=None, priority_func = get_priority, priority_type = "new_agent"):
    if graph_for_priority == None:
        graph_for_priority = graph
    
    W = set()
    A = set([seller])
    m = item_count
    P = set()
    neighbors_visited = set()
    visited = set()
    payments = {}

    while(True):
        explore_set = W | (A-P)
        for i in explore_set:
            if i not in neighbors_visited:
                A = A | set(graph[i])
                visited = visited | set(graph[i])
                neighbors_visited.add(i)
        A.discard(seller)

        E = A - W
        E_valuations = {i : valuations[i] for i in E}
        buyers_sort_in_E = sorted(E_valuations.items(), key=lambda item: item[1], reverse=True)
        top_m_buyers_in_E = buyers_sort_in_E[:m]
        P = W | set([pair[0] for pair in top_m_buyers_in_E])
        prioritys = {i : priority_func(i, graph_for_priority, priority_type, visited, seller) for i in (P-W)}
        w = max(prioritys, key=prioritys.get)
        W.add(w)
        payments[w] = 0 if len(buyers_sort_in_E)<m+1 else buyers_sort_in_E[m][1]
        m -= 1

        if P == W:
            break
    
    allocation = W
    SW = sum(valuations[i] for i in allocation)
    seller_revenue = sum(payments.values())
    return allocation, payments, SW, seller_revenue


if __name__ == "__main__":
    item_count = 10
    seller_count = 20
    graph_files = ["email-Eu-core", 
                   "facebook_combined",
                   "petster-friendships-hamster-uniq"]
    results = {key: 0 for key in graph_files}
    for file in graph_files:
        graph = dataset.get_graph('./dataset/'+file+'.txt')
        valuations = dataset.generate_valuations(graph, item_count, "hom_single")
        dataset.draw_valuations_distribution(valuations, f'./figure/hom_single/{file}_valuations.png')
        sellers = dataset.generate_sellers(graph, seller_count)
        print(file, sellers)
        mechanisms = [MPA_in_all, MPA_in_neighbors, VCG, VCG_RM, DNA_MU_R, MUDAN]
        result = {key: {"SW":[], "seller_revenue":[]} for key in mechanisms}

        for seller in sellers:
            for mechanism in mechanisms:
                allocation, payments, SW, seller_revenue = mechanism(graph, valuations, item_count, seller)
                result[mechanism]["SW"].append(SW)
                result[mechanism]["seller_revenue"].append(seller_revenue)
                print(seller, mechanism.__name__)
        results[file] = result

    color_map = plt.get_cmap('tab10')
    mechanism_count = len(mechanisms)
    for file in graph_files:
        fig, axes = plt.subplots(2, mechanism_count, figsize=(3 * mechanism_count, 5))
        fig.suptitle(f'{file}, item = {item_count}', fontsize=16)
        sw_min = []
        sw_max = []
        sr_min = []
        sr_max = []

        for i in range(mechanism_count):
            # 绘制 SW 折线图
            color = color_map(i % color_map.N)
            sw_data = results[file][mechanisms[i]]["SW"]
            axes[0, i].scatter(range(len(sw_data)),sw_data, color=color, label=mechanisms[i].__name__)
            axes[0, i].set_title(f'{mechanisms[i].__name__} - SW')
            axes[0, i].set_xlabel('Seller Index')
            axes[0, i].set_ylabel('SW')

            sw_min.append(min(sw_data))
            sw_max.append(max(sw_data))

            # 绘制 seller_revenue 折线图
            sr_data = results[file][mechanisms[i]]["seller_revenue"]
            axes[1, i].scatter(range(len(sr_data)),sr_data, color=color, label=mechanisms[i].__name__)
            axes[1, i].set_title(f'{mechanisms[i].__name__} - Seller Revenue')
            axes[1, i].set_xlabel('Seller Index')
            axes[1, i].set_ylabel('Seller Revenue')

            sr_min.append(min(sr_data))
            sr_max.append(max(sr_data))

        global_sw_min = min(sw_min)
        global_sw_max = max(sw_max)
        global_sr_min = min(sr_min)
        global_sr_max = max(sr_max)

        for i in range(mechanism_count):
            axes[0,i].set_ylim(global_sw_min-10, global_sw_max+10)
            axes[1,i].set_ylim(global_sr_min-10, global_sr_max+10)

        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(f'./figure/hom_single/{file}_plot.png')
        plt.close()