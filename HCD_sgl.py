# -*- coding:utf-8 -*-
"""
@Date: 2020-06-25
@Author: Tiko
@Email: twx@bupt.edu.cn
@Title: HCD_sgl
@Description: 基于单条原路径实现的异质信息网络社团划分，参考论文《异质信息网络中基于元路径的社团发现算法研究》
"""
import time
import copy
import numpy as np
import networkx as nx
from pysclump import PathSim
import matplotlib.pyplot as plt
import community as community_louvain

seedNum = 3  # 种子节点的数量
v = 1  # 每个节点最多有v个标签
type_lists = {
    'A': [],  # 作者节点
    'C': []  # 会议节点
}
incidence_matrices = { 
    "AC": None,
    "CA": None
}

# 加载网络
def load_graph(path):
    G = nx.Graph()
    shapes = ['o','D','v','^']
    with open(path + '/nodes') as text:
        for line in text:
            node = line.strip().split(" ")
            G.add_node(int(node[0]), shape=shapes[int(node[1])-1], tag=0)
            type_lists[node[2]].append(int(node[0]))
    with open(path + '/edges') as text:
        for line in text:
            vertices = line.strip().split(" ")
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target, weight=int(vertices[2]))
    return G

# 异质网络映射
# 将异质化网络通过矩阵相乘的方法映射为同质化网络
def transfor_graph(G):
    # 初始化矩阵
    incidence_matrices["AC"] = np.zeros((len(type_lists['A']), len(type_lists['C'])))
    incidence_matrices["CA"] = np.zeros((len(type_lists['C']), len(type_lists['A'])))
    # 更新矩阵内容
    for i in range(len(type_lists['A'])):
        A_node = type_lists['A'][i]
        for j in range(len(type_lists['C'])):
            C_node = type_lists['C'][j]
            if G.has_edge(A_node, C_node):
                incidence_matrices["AC"][i][j] = int(G.edges[A_node, C_node]['weight'])
                incidence_matrices["CA"][j][i] = int(G.edges[A_node, C_node]['weight'])
    # 将两个矩阵相乘得到同质化网络的邻接矩阵
    new_matrix = np.dot(incidence_matrices["AC"], incidence_matrices["CA"])
    # 将邻接矩阵转化为nx网络
    N_graph = nx.from_numpy_matrix(new_matrix)  
    return N_graph


# 克隆网络
def clone_graph(G):
    cloned_graph = nx.Graph()
    for edge in G.edges():
        cloned_graph.add_edge(edge[0], edge[1])
    return cloned_graph

# 求节点的邻居节点度之和
def get_kn(G, node):
    kn = 0
    for i in G.neighbors(node):
        kn = kn + G.degree(i, weight='weight')
    return kn

# 获取排序后的节点列表
def get_node_list(G):
    node_list = []
    for i in G.nodes():
        node_list.append([i, G.degree(i, weight='weight'), get_kn(G, i)])  # [节点名称，节点度，邻居节点度之和]
    return sorted(node_list, key=lambda x:(-x[1], x[2]))  # 按节点度从大到小排序，并按本地向心性由大到小（邻居节点度之和由小到大）排序

# 计算模块度
def cal_Q(partition, G):
    m = len(list(G.edges()))
    a = []
    e = []

    # 计算每个社区的a值
    for community in partition:
        t = 0
        for node in community:
            t += len(list(G.neighbors(node)))
        a.append(t / float(2 * m))

    # 计算每个社区的e值
    for community in partition:
        t = 0
        for i in range(len(community)):
            for j in range(len(community)):
                if i != j:
                    if G.has_edge(community[i], community[j]):
                        t += 1
        e.append(t / float(2 * m))

    # 计算Q
    q = 0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q

# 图可视化
def showGraph(G, title="Result"):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    pos = nx.spring_layout(G)
    # 获取边的权重
    edge_labels = nx.get_edge_attributes(G,'weight')
    # 可视化节点，在不同的地方节点的属性可能不同
    try:
        for i in G.nodes():
            nx.draw_networkx_nodes(G, pos, nodelist = [i], 
                node_shape = G.node[i]['shape'],
                node_color = colors[G.node[i]['tag']],
                node_size = 350,
                alpha = 1)
    except KeyError:
        try:
            for i in G.nodes():
                nx.draw_networkx_nodes(G, pos, nodelist = [i], 
                    node_color = colors[G.node[i]['tag']],
                    node_size = 350,
                    alpha = 1)
        except KeyError:
            for i in G.nodes():
                nx.draw_networkx_nodes(G, pos, nodelist = [i], 
                    node_size = 350,
                    alpha = 1)
    # 可视化边
    nx.draw_networkx_edges(G, pos, width = 1, alpha = 1)
    # 可视化label
    nx.draw_networkx_labels(G, pos, font_size = 12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

# 找到列表中最大值所在的位置
def findmax(s_list):
    t = -1
    max = -9999
    for i in range(len(s_list)):
        if s_list[i] > max:
            t = i
            max = s_list[i]
    return t

if __name__ == '__main__':
    # 加载文件中的网络并可视化
    G = load_graph(".")
    G_c = clone_graph(G)  # 复制一份网络以备不时之需 
    # # 可视化展示
    # showGraph(G)
    N_G = transfor_graph(G)

    # 一、初始社团划分
    # 1. 获取种子节点
    seedNode = []
    node_list = get_node_list(N_G)[:seedNum]
    for node_info in node_list:
        seedNode.append(node_info[0])
    # print(get_node_list(N_G))

    # 2. 确定种子节点的标签
    # 此处应该先进行社团检测，然后重构网络，也就是louvain(fast folding)算法，我们用community库中的函数（https://github.com/taynaud/python-louvain）来实现。
    partition = community_louvain.best_partition(N_G)
    # print(partition)
    for node in N_G.nodes():
        if node in seedNode:
            N_G.node[node]['tag'] = partition[node] + 1
            G.node[node+1]['tag'] = partition[node] + 1
        else:
            N_G.node[node]['tag'] = 0
            G.node[node+1]['tag'] = 0
    # # 可视化展示
    # showGraph(G)

    # 3. 确定非种子节点的标签
    # 获取基于元路径APA（作者-论文-作者）的节点相似性矩阵，使用PathSim算法
    # Create PathSim instance.
    ps = PathSim(type_lists, incidence_matrices)

    # Get the similarity matrix M for the metapath.
    sim_matrix = ps.compute_similarity_matrix(metapath='ACA')

    # 遍历全部的非种子节点，判断其邻居节点中是否存在种子节点。
    # 若该非种子节点的邻居中存在种子节点，则先获取它的全部种子节点邻居所属的社团集合，
    # 再计算非种子节点属于各个社团的可能性，最后将可能性最大的那个社团标签作为非种子节点的社团标签。
    for node in N_G.nodes():
        if node not in seedNode:
            seed_neibors_sim = []
            seed_neibors=[]
            for neibor in N_G.neighbors(node):
                if neibor in seedNode:
                    seed_neibors_sim.append(sim_matrix[node][neibor])
                    seed_neibors.append(neibor)
            # 邻居中存在种子节点
            if len(seed_neibors_sim) > 0:
                close_node = seed_neibors[findmax(seed_neibors_sim)] + 1
                G.node[node+1]['tag'] = G.node[close_node]['tag']
            # 邻居中不存在种子节点
            else:
                for s_node in seedNode:
                    seed_neibors_sim.append(sim_matrix[node][s_node])
                close_node = seedNode[findmax(seed_neibors_sim)] + 1
                G.node[node+1]['tag'] = G.node[close_node]['tag']
    # 可视化展示
    showGraph(N_G, title="Preliminary classification results")

    # 初始化m_t_1
    m_t_1 = {}
    for A_node in type_lists['A']:
        if G.node[A_node]['tag'] not in m_t_1:
            m_t_1[G.node[A_node]['tag']] = 1  # 此处C0中的tag以1开头
        else:
            m_t_1[G.node[A_node]['tag']] = m_t_1[G.node[A_node]['tag']] + 1


    # 二、最终社团划分

    # 初始化b_t-1矩阵
    b_t_1 = np.zeros((max(m_t_1.keys()), len(type_lists['A'])))  # 矩阵中的C0维度应该使用“坐标+1”作为真正的tag值
    for node in type_lists['A']:
        b_t_1[G.node[node]["tag"]-1][node-1] = 1

    # 初始化m_t
    m_t = m_t_1.copy()
    for i in m_t:
        m_t[i] = None
    
    b_t = b_t_1.copy()
    while m_t != m_t_1:
        if None not in m_t.values():
            m_t_1 = m_t.copy()
        b_t_1 = b_t.copy()
        for i in range(len(type_lists["A"])):
            b_max = 0
            c_max = G.node[type_lists["A"][i]]["tag"]
            # 根据公式(3)更新b_t
            for c in m_t.keys():  # 此处m_t中的keys应该储存着当前存在的所有社团标签，遍历这些标签求b(c, i)
                b_t_num = 0  # 公式3中的分子
                b_t_den = 0  # 公式3中的分母
                for neibor_No in N_G.neighbors(type_lists["A"][i] - 1):
                    b_t_num = b_t_num + b_t_1[c-1][neibor_No] * sim_matrix[type_lists["A"][i]-1][neibor_No]  # (c,x)的保存位置应该是[c-1][x-1]
                    b_t_den = b_t_den + sim_matrix[type_lists["A"][i]-1][neibor_No]
                b_t[c-1][type_lists["A"][i]-1] = b_t_num/b_t_den
            for c_i in range(len(b_t[:, type_lists["A"][i]-1])):
                t = b_t[c_i][type_lists["A"][i]-1]
                if t < 1/v:
                    if t > b_max:
                        b_max = t
                        c_max = c_i
                    b_t[c_i][type_lists["A"][i]-1] = 0
            if max(b_t[:, type_lists["A"][i]-1]) == 0:
                b_t[c_max][type_lists["A"][i]-1] = 1
            else:
                for c_i in range(len(b_t[:, type_lists["A"][i]-1])):
                    b_t[c_i][type_lists["A"][i]-1] = b_t[c_i][type_lists["A"][i]-1] / sum(b_t[:, type_lists["A"][i]-1])
        # 按照公式(4)更新m_t
        m_t_tmp = {}
        m_t = {}
        for c_i in range(len(b_t)):
            m_t_tmp[c_i+1] = sum(b_t[c_i])
        for i in m_t_1.keys():
            m_t[i] = int(min([m_t_tmp[i], m_t_1[i]]))
    # 更新节点的标签
    for node in type_lists["A"]:
        G.node[node]['tag'] = findmax(b_t[:, node-1]) + 1


    # 根据算法结果画图
    showGraph(G)