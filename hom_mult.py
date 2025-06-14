from collections import deque
import copy
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import random


class Problem:
    def __init__(self, graph: dict, valuations: dict, item_count: int, seller: int):
        self.graph = graph
        self.valuations = valuations
        self.item_count = item_count
        self.seller = seller

class Result:
    def __init__(self, allocation: list, payments: dict, SW: int, Rev: int):
        self.allocation = allocation
        self.payments = payments
        self.SW = SW
        self.Rev = Rev


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


def generate_valuations(graph: dict, item_count: int):
    valuations = {}
    for i in graph:
        demand_count = random.randint(1, item_count)
        valuation = [random.randint(1, 1000) for _ in range(demand_count)]
        valuation.sort(reverse=True)
        valuations[i] = valuation
    return valuations


def generate_sellers(graph: dict, seller_count: int):
    return random.sample(list(graph.keys()), seller_count)


def find_reachable(graph, seller):
    visited = []
    queue = deque([seller])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            neighbors = graph.get(node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
    return visited


def build_dominator_tree(graph: dict, seller: int):
    """构建支配树 - 使用简化的算法"""
    
    # 首先找到所有从seller可达的节点
    reachable = find_reachable(graph, seller)
    
    # 初始化支配关系
    dominators = {node: set(reachable) for node in reachable}
    dominators[seller] = {seller}
    
    # 迭代计算支配关系直到收敛
    changed = True
    while changed:
        changed = False
        for node in reachable:
            if node == seller:
                continue
                
            # 找到所有能到达当前节点的前驱
            predecessors = []
            for pred in reachable:
                if node in graph.get(pred, set()):
                    predecessors.append(pred)
            
            if not predecessors:
                continue
                
            # 新的支配者集合是所有前驱支配者集合的交集，再加上节点自己
            new_dominators = set([node])
            if predecessors:
                # 计算所有前驱的支配者交集
                intersection = dominators[predecessors[0]].copy()
                for pred in predecessors[1:]:
                    intersection &= dominators[pred]
                new_dominators |= intersection
            
            # 检查是否有变化
            if new_dominators != dominators[node]:
                dominators[node] = new_dominators
                changed = True
    
    # 构建立即支配者关系
    idom = {}
    for node in reachable:
        if node == seller:
            continue
        
        # 移除节点自己，得到真正的支配者
        node_dominators = dominators[node] - {node}
        
        # 找到立即支配者（移除被其他支配者支配的支配者）
        immediate_dom = None
        for dom in node_dominators:
            is_immediate = True
            for other_dom in node_dominators:
                if dom != other_dom and dom in dominators[other_dom]:
                    is_immediate = False
                    break
            if is_immediate:
                immediate_dom = dom
                break
        
        if immediate_dom is not None:
            idom[node] = immediate_dom
    
    # 构建支配树
    dominator_tree = {node: [] for node in reachable}
    dominator_tree_reverse = {}
    
    for node, immediate_dominator in idom.items():
        if immediate_dominator in dominator_tree:
            dominator_tree[immediate_dominator].append(node)
            dominator_tree_reverse[node] = immediate_dominator
    
    return dominator_tree, dominator_tree_reverse


def build_BFS_tree(graph, seller):
    # 初始化BFS树和队列
    bfs_tree = {}
    visited = set([seller])
    queue = deque([seller])
    
    # BFS遍历构建树
    while queue:
        node = queue.popleft()
        bfs_tree[node] = []
        neighbors = graph[node]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                bfs_tree[node].append(neighbor)
    return bfs_tree

def VCG_SN(problem: Problem, mu):
    graph = problem.graph
    valuations = problem.valuations
    seller = problem.seller
    item_count = problem.item_count

    buyers = find_reachable(graph, seller)
    buyers.remove(seller)
    IDT, IDT_reverse = build_dominator_tree(graph, seller) 
    payments = {}

    buyers_valuations = {}
    for i in buyers:
        demand_count = len(valuations[i])
        for j in range(demand_count):
            buyers_valuations[(i,j)] = valuations[i][j]
    top_m_buyers = sorted(buyers_valuations.items(), key=lambda item: item[1], reverse=True)[:item_count]
    allocation = {}
    for buyer, valuation in top_m_buyers:
        i, j = buyer
        if i not in allocation:
            allocation[i] = 1
        else:
            allocation[i] += 1
    optimal_SW = sum(pair[1] for pair in top_m_buyers)
    
    critical_nodes = set()
    for i in allocation:
        current = i
        while (current != seller) and (current in IDT_reverse):
            critical_nodes.add(current)
            current = IDT_reverse[current]
    for i in critical_nodes:
        subtree_i = find_reachable(IDT, i)
        buyers_except_i = [buyer for buyer in buyers if buyer not in subtree_i]
        valuations_except_i = {}
        for j in buyers_except_i:
            demand_count = len(valuations[j])
            for k in range(demand_count):
                valuations_except_i[(j, k)] = valuations[j][k]
        top_m_buyers_except_i = sorted(valuations_except_i.items(), key=lambda item: item[1], reverse=True)[:item_count]
        optimal_SW_except_i = sum(pair[1] for pair in top_m_buyers_except_i)
        item_allocated = 0 if i not in allocation else allocation[i]
        others_SW_except_i = optimal_SW - sum(valuations[i][:item_allocated])
        payments[i] = optimal_SW_except_i - others_SW_except_i
    Rev = sum(payments.values())
    return Result(allocation, payments, optimal_SW, Rev)


def VCG_IV(problem: Problem, mu):
    graph = problem.graph
    valuations = problem.valuations
    seller = problem.seller
    item_count = problem.item_count

    remain_item = item_count
    buyers = [seller]
    visited = set([seller])
    payments = {}
    allocation = {}
    SW = 0
    
    while remain_item > 0:
        new_buyers = []
        for i in buyers:
            neighbors = graph[i]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_buyers.append(neighbor)
        buyers = new_buyers
        if not buyers:
            break

        buyers_valuations = {}
        for i in buyers:
            demand_count = len(valuations[i])
            for j in range(demand_count):
                buyers_valuations[(i,j)] = valuations[i][j]
        buyers_sorted = sorted(buyers_valuations.items(), key=lambda item: item[1], reverse=True)
        top_m_buyers = buyers_sorted[:remain_item]
        tmp_SW = sum(pair[1] for pair in top_m_buyers)
        for buyer, valuation in top_m_buyers:
            i, j = buyer
            if i not in allocation:
                allocation[i] = 1           
            else:
                allocation[i] += 1
        
        for i in buyers:
            if i in allocation:
                buyers_sorted_except_i = [buyer for buyer in buyers_sorted if buyer[0][0] != i]
                optimal_SW_except_i = sum(valuation for (j, k), valuation in buyers_sorted_except_i[:remain_item])
                other_SW_except_i = tmp_SW - sum(valuations[i][:allocation[i]])
                payments[i] = optimal_SW_except_i - other_SW_except_i
        
        SW += tmp_SW
        remain_item -= len(top_m_buyers)
    
    Rev = sum(payments.values())
    return Result(allocation, payments, SW, Rev)

def LDM(problem: Problem, mu = 1e5):
    graph = problem.graph
    valuations = problem.valuations
    seller = problem.seller
    item_count = problem.item_count

    bfs_tree = build_BFS_tree(graph, seller)
    remain_item = item_count
    buyers = set([seller])
    payments = {}
    allocation = {}
    SW = 0

    while remain_item > 0:
        new_buyers = set()
        for i in buyers:
            new_buyers |= set(bfs_tree[i])
        buyers = new_buyers
        if not buyers:
            break

        agents = copy.deepcopy(buyers)
        for i in buyers:
            neighbors_with_children = [neighbor for neighbor in bfs_tree[i] if bfs_tree[neighbor]]
            neighbors = [neighbor for neighbor in bfs_tree[i] if not bfs_tree[neighbor]]  
            neighbor_valuations = {neighbor: valuations[neighbor][0] for neighbor in neighbors}
            neighbor_sorted = sorted(neighbor_valuations.items(), key=lambda item: item[1], reverse=True)
            neighbors = [pair[0] for pair in neighbor_sorted[item_count+mu-len(neighbors_with_children):]]
            agents |= set(neighbors)

        agent_valuations = {}
        for i in agents:
            demand_count = len(valuations[i])
            for j in range(demand_count):
                agent_valuations[(i,j)] = valuations[i][j]
        agents_sorted = sorted(agent_valuations.items(), key=lambda item: item[1], reverse=True)
        top_m_agents = agents_sorted[:remain_item]
        optimal_SW = sum(pair[1] for pair in top_m_agents)
        new_remain_item = remain_item
        for agent, valuation in top_m_agents:
            i, j = agent
            if i in buyers:
                SW += valuation
                new_remain_item -= 1
                if i not in allocation:
                    allocation[i] = 1          
                else:
                    allocation[i] += 1
        
        for i in buyers:
            agents_except_i = [agent for agent in agents if agent != i and agent not in bfs_tree[i]]          
            agent_valuations_except_i = {}
            for j in agents_except_i: 
                demand_count = len(valuations[j])
                for k in range(demand_count):
                    agent_valuations_except_i[(j,k)] = valuations[j][k]
            
            agents_sorted_except_i = sorted(agent_valuations_except_i.items(), key=lambda item: item[1], reverse=True)
            top_m_agents_except_i = agents_sorted_except_i[:remain_item]
            optimal_SW_except_i = sum(pair[1] for pair in top_m_agents_except_i)
            item_allocated = 0 if i not in allocation else allocation[i]
            others_SW_except_i = optimal_SW - sum(valuations[i][:item_allocated])
            payments[i] = optimal_SW_except_i - others_SW_except_i
        remain_item = new_remain_item

    Rev = sum(payments.values())
    
    return Result(allocation, payments, SW, Rev)    


def MUDAN_m(problem: Problem, mu):
    graph = problem.graph
    valuations = problem.valuations
    seller = problem.seller
    item_count = problem.item_count

    prioritys = {i: len(graph[i]) for i in graph}
    W = set()
    A = set([seller])
    m = item_count
    P = set()
    visited = set([seller])
    payments = {}
    valuations = {i: valuations[i]+[0 for _ in range(item_count-len(valuations[i]))] for i in valuations}

    while(True):
        if seller in A:
            A = set([(k, 0) for k in graph[seller]])
        else:
            explore_set = W | (A-P)
            for (i,j) in explore_set:
                if j != item_count-1: 
                    A.add((i, j+1))
                else:
                    if i not in visited:
                        A = A | set([(k, 0) for k in graph[i]])              
                        visited.add(i)
            A. discard(seller)

        E = A - W
        E_valuations = {(i, j) : valuations[i][j] for (i, j) in E}
        buyers_sort_in_E = sorted(E_valuations.items(), key=lambda item: item[1], reverse=True)
        top_m_buyers_in_E = buyers_sort_in_E[:m]
        P = W | set([pair[0] for pair in top_m_buyers_in_E])
        P_prioritys = {(i, j) : prioritys[i] for (i,j) in (P-W)}
        w = max(P_prioritys, key=P_prioritys.get)
        W.add(w)
        payments[w] = 0 if len(buyers_sort_in_E)<m+1 else buyers_sort_in_E[m][1]
        m -= 1

        if P == W:
            break
    
    allocation = {}
    for (i, j) in W:
        if i not in allocation:
            allocation[i] = 1
        else:
            allocation[i] += 1

    new_payments = {}
    for (i,j), payment in payments.items():
        if i not in new_payments:
            new_payments[i] = payment
        else:
            new_payments[i] += payment

    SW = sum(valuations[i][j] for (i,j) in W)
    Rev = sum(new_payments.values())
    return Result(allocation, new_payments, SW, Rev)


def Refined_MUDAN_m(problem: Problem, mu):
    graph = problem.graph
    valuations = problem.valuations
    seller = problem.seller
    item_count = problem.item_count

    prioritys = {i: len(graph[i]) for i in graph}
    W = set()
    A = set([seller])
    m = item_count
    P = set()
    visited = set([seller])
    payments = {}
    num = 0
    while(True):
        num+=1
        if seller in A:
            A = set([(k, 0) for k in graph[seller]])
        else:
            explore_set = A - P
            for (i,j) in W:         
                if j != len(valuations[i])-1: 
                    A.add((i, j+1))
                else:
                    if i not in visited:
                        A = A | set([(k, 0) for k in graph[i]])              
                        visited.add(i)
            for (i,j) in explore_set:
                if i not in visited:
                    A = A | set([(k, 0) for k in graph[i]])
                    visited.add(i)
            A. discard((seller, 0))

        E = A - W
        E_valuations = {(i, j) : valuations[i][j] for (i, j) in E}
        buyers_sort_in_E = sorted(E_valuations.items(), key=lambda item: item[1], reverse=True)
        top_m_buyers_in_E = buyers_sort_in_E[:m]
        P = W | set([pair[0] for pair in top_m_buyers_in_E])
        P_prioritys = {(i, j) : prioritys[i] for (i,j) in (P-W)}
        w = max(P_prioritys, key=P_prioritys.get)
        W.add(w)
        payments[w] = 0 if len(buyers_sort_in_E)<m+1 else buyers_sort_in_E[m][1]
        m -= 1

        if P == W:
            break
    
    allocation = {}
    for (i, j) in W:
        if i not in allocation:
            allocation[i] = 1
        else:
            allocation[i] += 1

    new_payments = {}
    for (i,j), payment in payments.items():
        if i not in new_payments:
            new_payments[i] = payment
        else:
            new_payments[i] += payment

    SW = sum(valuations[i][j] for (i,j) in W)
    Rev = sum(new_payments.values())
    return Result(allocation, new_payments, SW, Rev)


if __name__ == "__main__":
    item_count = 10
    seller_count = 20
    graph_files = [
                   "email-Eu-core", 
                   "facebook_combined",
                   "petster-friendships-hamster-uniq"
                   ]
    mu = {
        "email-Eu-core": 150, 
        "facebook_combined": 60,
        "petster-friendships-hamster-uniq": 130
         }
    
    for file in graph_files:
        graph = get_graph('./dataset/'+file+'.txt')
        valuations = generate_valuations(graph, item_count)
        sellers = generate_sellers(graph, seller_count)
        print(file, sellers)
        mechanisms = [VCG_SN, VCG_IV, LDM, MUDAN_m, Refined_MUDAN_m]
        result = {key: {"SW":[], "Rev":[]} for key in mechanisms}

        for seller in sellers:
            problem = Problem(graph, valuations, item_count, seller)
            for mechanism in mechanisms:
                mechanism_result = mechanism(problem, mu[file])
                result[mechanism]["SW"].append(mechanism_result.SW)
                result[mechanism]["Rev"].append(mechanism_result.Rev)
                print(seller, mechanism.__name__)

        # 生成当前file的统计图
        color_map = plt.get_cmap('tab10')
        mechanism_count = len(mechanisms)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'{file}, item = {item_count}', fontsize=16)
        
        # 设置横轴偏移量，让不同机制的点错开
        offset_width = 0.15
        offsets = [(i - mechanism_count/2 + 0.5) * offset_width for i in range(mechanism_count)]
        
        sw_min = []
        sw_max = []
        sr_min = []
        sr_max = []

        for i in range(mechanism_count):
            color = color_map(i % color_map.N)
            mechanism_name = mechanisms[i].__name__
            
            # 绘制 SW 散点图
            sw_data = result[mechanisms[i]]["SW"]
            x_positions = [j + offsets[i] for j in range(len(sw_data))]
            axes[0].scatter(x_positions, sw_data, color=color, label=mechanism_name, alpha=0.7)
            sw_min.append(min(sw_data))
            sw_max.append(max(sw_data))

            # 绘制 Rev 散点图
            sr_data = result[mechanisms[i]]["Rev"]
            axes[1].scatter(x_positions, sr_data, color=color, label=mechanism_name, alpha=0.7)
            sr_min.append(min(sr_data))
            sr_max.append(max(sr_data))

        # 添加竖直虚线分隔不同卖家
        for seller_idx in range(1, len(sellers)):
            axes[0].axvline(x=seller_idx - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            axes[1].axvline(x=seller_idx - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        # 设置图表属性
        axes[0].set_title('Social Welfare (SW)')
        axes[0].set_xlabel('Seller Index')
        axes[0].set_ylabel('SW')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

        axes[1].set_title('Seller Revenue')
        axes[1].set_xlabel('Seller Index')
        axes[1].set_ylabel('Seller Revenue')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

        # 设置y轴范围
        global_sw_min = min(sw_min)
        global_sw_max = max(sw_max)
        global_sr_min = min(sr_min)
        global_sr_max = max(sr_max)

        axes[0].set_ylim(0, global_sw_max*1.1)
        axes[1].set_ylim(global_sr_min-10, global_sr_max*1.1)

        # 设置x轴刻度
        axes[0].set_xticks(range(len(sellers)))
        axes[1].set_xticks(range(len(sellers)))

        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(f'./figure/hom_mult/{file}_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated plot for {file}")