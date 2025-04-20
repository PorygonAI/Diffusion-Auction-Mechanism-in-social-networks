import hom_single
import dataset
import matplotlib.pyplot as plt


def Generalized_graph_and_valuations(graph: dict, valuations: dict, item_count: int, seller: int):
    generalized_graph = {}
    generalized_valuations = {}

    generalized_graph[seller] = set()
    for i in graph:
        if i != seller:
            for j in range(item_count):
                new_buyer = (i, j)
                generalized_graph[new_buyer] = set()
                generalized_valuations[new_buyer] = valuations[i][j]

    for neighbor in graph[seller]:
        generalized_graph[seller].add((neighbor, 0))

    for i in graph:
        if i != seller:
            for j in range(item_count-1):
                generalized_graph[(i, j)].add((i, j+1))
            for neighbor in graph[i]:
                if neighbor != seller:
                    generalized_graph[(i, item_count-1)].add((neighbor, 0))

    return generalized_graph, generalized_valuations


def get_priority_mult(i: int, graph: dict, priority_type, visited: set, seller: int):
    new_visited = set()
    for j in visited:
        if isinstance(j, tuple):
            new_visited.add(j[0])   
        else:
            new_visited.add(seller) 
    return hom_single.get_priority(i[0], graph, priority_type, new_visited, seller)


def MUDAN_m(graph: dict, valuations: dict, item_count: int, seller: int, priority_type = "new_agent"):
    generalized_graph, generalized_valuations = Generalized_graph_and_valuations(graph, valuations, item_count, seller)
    MUDAN_allocation, MUDAN_payments, SW, seller_revenue = hom_single.MUDAN(generalized_graph, generalized_valuations, item_count, seller, graph, get_priority_mult, priority_type)
    allocation = {}
    payments =  {}
    for i in MUDAN_payments:
        if i[0] in payments:
            allocation[i[0]] += 1
            payments[i[0]] += MUDAN_payments[i]
        else:
            allocation[i[0]] = 1
            payments[i[0]] = MUDAN_payments[i]
    return allocation, payments, SW, seller_revenue


def VCG_in_all(graph: dict, valuations: dict, item_count: int, seller: int):
    buyers = hom_single.find_reachable(graph, seller, None)
    buyers_valuations = {}
    for i in buyers:
        for j in range(item_count):
            new_buyer = (i, j)
            buyers_valuations[new_buyer] = valuations[i][j]

    buyers_sorted = sorted(buyers_valuations.items(), key=lambda item: item[1], reverse=True)
    top_m_buyers = buyers_sorted[:item_count]
    optimal_SW = sum(pair[1] for pair in top_m_buyers)
    
    allocation = {}
    for buyer, valuation in top_m_buyers:
        i, j = buyer
        if i not in allocation:
            allocation[i] = 1
        else:
            allocation[i] += 1
    
    payments = {}
    for i in allocation:
        buyers_sorted_expect_i = [(buyer, valuation) for buyer, valuation in buyers_sorted if buyer[0] != i]
        top_m_buyers_expect_i = buyers_sorted_expect_i[:item_count]
        SW_except_i = sum(pair[1] for pair in top_m_buyers_expect_i)
        SW_i = sum(valuations[i][j] for j in range(allocation[i]))
        payments[i] = SW_except_i - (optimal_SW - SW_i)

    seller_revenue = sum(payments.values())
    return allocation, payments, optimal_SW, seller_revenue


def VCG_in_neighbors(graph: dict, valuations: dict, item_count: int, seller: int):
    buyers = graph[seller]
    buyers_valuations = {}
    for i in buyers:
        for j in range(item_count):
            new_buyer = (i, j)
            buyers_valuations[new_buyer] = valuations[i][j]

    buyers_sorted = sorted(buyers_valuations.items(), key=lambda item: item[1], reverse=True)
    top_m_buyers = buyers_sorted[:item_count]
    optimal_SW = sum(pair[1] for pair in top_m_buyers)
    
    allocation = {}
    for buyer, valuation in top_m_buyers:
        i, j = buyer
        if i not in allocation:
            allocation[i] = 1
        else:
            allocation[i] += 1
    
    payments = {}
    for i in allocation:
        buyers_sorted_expect_i = [(buyer, valuation) for buyer, valuation in buyers_sorted if buyer[0] != i]
        top_m_buyers_expect_i = buyers_sorted_expect_i[:item_count]
        SW_except_i = sum(pair[1] for pair in top_m_buyers_expect_i)
        SW_i = sum(valuations[i][j] for j in range(allocation[i]))
        payments[i] = SW_except_i - (optimal_SW - SW_i)

    seller_revenue = sum(payments.values())
    return allocation, payments, optimal_SW, seller_revenue


if __name__ == "__main__":
    item_count = 10
    seller_count = 20
    graph_files = ["email-Eu-core", 
                   "facebook_combined",
                   "petster-friendships-hamster-uniq"]
    results = {key: 0 for key in graph_files}
    for file in graph_files:
        graph = dataset.get_graph('./dataset/'+file+'.txt')
        valuations = dataset.generate_valuations(graph, item_count, "hom_mult")
        sellers = dataset.generate_sellers(graph, seller_count)
        print(file, sellers)
        mechanisms = [VCG_in_all, VCG_in_neighbors]
        priority_types = ["constant", "new_agent", "degree", "distance", "negative_distance"]
        all_list = mechanisms + priority_types
        result = {key: {"SW":[], "seller_revenue":[]} for key in all_list}

        for seller in sellers:
            for mechanism in mechanisms:
                    allocation, payments, SW, seller_revenue = mechanism(graph, valuations, item_count, seller)
                    result[mechanism]["SW"].append(SW)
                    result[mechanism]["seller_revenue"].append(seller_revenue)
                    print(seller, mechanism.__name__)
            for priority_type in priority_types:
                allocation, payments, SW, seller_revenue = MUDAN_m(graph, valuations, item_count, seller, priority_type)
                result[priority_type]["SW"].append(SW)
                result[priority_type]["seller_revenue"].append(seller_revenue)
                print(seller, priority_type)
        results[file] = result

    color_map = plt.get_cmap('tab10')
    all_count = len(all_list)
    mechanism_count = len(mechanisms)
    priority_count = len(priority_types)
    for file in graph_files:
        fig, axes = plt.subplots(2, all_count, figsize=(3 * all_count, 5))

        sw_min = []
        sw_max = []
        sr_min = []
        sr_max = []

        for i in range(all_count):
            # 绘制 SW 折线图
            color = color_map(i % color_map.N)
            sw_data = results[file][all_list[i]]["SW"]
            axes[0, i].plot(sw_data, color=color)
            if i < mechanism_count:
                axes[0, i].set_title(f'{mechanisms[i].__name__} - SW')
            else:
                axes[0, i].set_title(f'{priority_types[i-mechanism_count]} - SW')
            axes[0, i].set_xlabel('Seller Index')
            axes[0, i].set_ylabel('SW')

            sw_min.append(min(sw_data))
            sw_max.append(max(sw_data))

            # 绘制 seller_revenue 折线图
            sr_data = results[file][all_list[i]]["seller_revenue"]
            axes[1, i].plot(sr_data, color=color)
            if i < mechanism_count:
                axes[1, i].set_title(f'{mechanisms[i].__name__} - Seller Revenue')
            else:
                axes[1, i].set_title(f'{priority_types[i-mechanism_count]} - Seller Revenue')           
            axes[1, i].set_xlabel('Seller Index')
            axes[1, i].set_ylabel('Seller Revenue')

            sr_min.append(min(sr_data))
            sr_max.append(max(sr_data))

        global_sw_min = min(sw_min)
        global_sw_max = max(sw_max)
        global_sr_min = min(sr_min)
        global_sr_max = max(sr_max)

        for i in range(all_list):
            axes[0,i].set_ylim(global_sw_min-10, global_sw_max+10)
            axes[1,i].set_ylim(global_sr_min-10, global_sr_max+10)

        plt.tight_layout()
        plt.savefig(f'./figure/hom_mult/{file}_plot.png')
        plt.close()

#     graph = {'s':['a','b','c','h'], 'a':['s','d','e'],'b':['s','e','f'], 'c':['s','f'],'d':['a','g'],'e':['a','b'],'f':['b','c'],'g':['d'],'h':['s']}
#     valuations = {'a':(10,9,8),'b':(6,5,0),'c':(7,6,3),'d':(5,3,2),'e':(4,2,1),'f':(9,7,0),'g':(12,11,0),'h':(2,1,1)}
    
#     MUDAN_m(graph, valuations, 3, 's', priority_type = 2)


