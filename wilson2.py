#!/usr/bin/env python3
"""
Wilson's algorithm for unweighted STs.
"""

import numpy as np
import networkx as nx
import sys
sys.path.append('/home/carlo2/workspace/networkqit')
import matplotlib.pyplot as plt
import networkqit as nq
import random
import matplotlib

class Wilson:

    def __init__(self, G, q):
        self.G = G
        self.H = nx.DiGraph()
        self.nv = G.number_of_nodes()
        self.q = q
        
        # set edge attribute weight with weight 1
        self.H.add_weighted_edges_from([(u,v,1.0) for u,v in G.edges()])
        self.H.add_weighted_edges_from([(v,u,1.0) for u,v in G.edges()])
        
        # add the magic link from all nodes to n+1 with weight 1/q
        #self.H.add_weighted_edges_from([(self.nv,n, q) for n in G.nodes()])
        self.H.add_weighted_edges_from([(n,self.nv, 1/q) for n in G.nodes()])
        #print(np.asarray([(u,v,w['weight']) for u,v,w in self.H.edges(data=True)]).reshape([60,3]))

    # Choose an edge from v's adjacency list (randomly)
    def random_successor(self, v):
        nei = list(self.H.neighbors(v))
        weight = np.array([ self.H.get_edge_data(v,u)['weight'] for u in nei], dtype=float)
        weight /= weight.sum()
        return np.random.choice(nei, p = weight)
    
    def sample(self):
        intree = [False] * (self.nv+1)
        successor = {}
        # put the additional node
        F = nx.DiGraph()
        self.roots = set()
        root = self.nv
        intree[root] = True
        successor[root] = None

        from random import shuffle
        l = [root] + list(range(self.nv))
        for i in l:
            u = i
            while not intree[u]:
                successor[u] = self.random_successor(u)
                if successor[u] == self.nv: # if the last node of the trajectory is ∆ add it to the roots
                    self.roots.add(u)
                u = successor[u]
            u = i # come back to the node it started from
            # remove self-loops
            while not intree[u]:
                intree[u] = True
                #if u in successor:
                u = successor[u]

        # Creates the random forest
        for i in range(self.nv):
            if i in successor.keys():
                neighbor = successor[i]
                if neighbor is not None:
                    F.add_edge(i,neighbor)
        
        if self.nv in self.roots:
            self.roots.remove(self.nv)
        # remove the magic node, together with all its links
        F.remove_node(F.number_of_nodes() - 1)
        #F.remove_edges_from([(u,self.nv,1.0) for u in F.nodes()])
        # save the leaves
        self.leaves = [n for n in F.nodes() if F.degree(n)==1]
        return F, list(self.roots)
    
    def s(self):
        L = nx.laplacian_matrix(self.G).toarray()
        lambdai = np.linalg.eigvalsh(L)
        return (self.q/(self.q + lambdai)).sum()

def to_networkx(G, T, root_nodes=None):
    T = nx.Graph(T)
    n_trees = nx.number_connected_components(T)
    pos = nx.kamada_kawai_layout(G)
    if root_nodes is not None:
        nx.draw_networkx_nodes(G, pos=pos, nodelist=root_nodes, node_color='r',node_size=25,linew_width=1)
        #nx.draw_networkx_labels(G, pos=pos, labels={i: i for i in range(G.number_of_nodes())})
    nx.draw_networkx_nodes(G, pos=pos, node_color='k', node_size=3, lines_width=0.1)
    nx.draw_networkx_edges(G, pos, edge_style='dashed', alpha=0.1, edge_color='k', edge_width=0.01)

    cmap = matplotlib.cm.get_cmap('viridis')
    for i, t in enumerate(nx.connected_component_subgraphs(T)):
        e = nx.number_of_edges(t)
        #print('|V|=%d |E|=%d' % (t.number_of_nodes(),t.number_of_edges()))
        nx.draw_networkx_edges(t, pos, width=2, edge_cmap=plt.cm.Set2, edge_color=[cmap(float(i)/n_trees)]*e )
    plt.axis('off')
    #plt.show()

# Controllare se beta è collegato alla massima distanza delle foglie dal nodo root
if __name__=='__main__':
    reps = 10
    G = nx.grid_2d_graph(20, 20, periodic=False)
    #G = nx.path_graph(100)
    G = nx.from_numpy_array(nx.to_numpy_array(G))
    
    #G = nx.planted_partition_graph(4,20,1,0.01)
    beta_range = np.logspace(-3, 3, 50)

    L = nx.laplacian_matrix(G).toarray()

    S = nq.entropy(L=L, beta_range=beta_range)
    Tr = np.log([np.sum(np.exp(-beta*np.linalg.eigvalsh(L))) for beta in beta_range])


    StreeQuenched = np.mean(np.log([[len(Wilson(G,q=q).sample()[1]) for _ in range(reps)] for q in beta_range]),1)
    #StreeAnnealed = np.log(np.mean(([[len(Wilson(G,q=q).sample()[1]) for q in beta_range] for _ in range(reps)]),0))
    plt.semilogx(beta_range, StreeQuenched, color='green', label='<log |R|>')
    #plt.semilogx(beta_range, StreeAnnealed, color='purple', label='log (<|R|>)')
    plt.semilogx(beta_range, np.log([Wilson(G,q=1/q).s() for q in beta_range]), color='magenta', label='stieltjes s(q)')
    plt.semilogx(beta_range, S, color='blue', label='S(\\rho)')
    #plt.semilogx(beta_range, Tr, color='red', label='Z')
    plt.legend()
    # plt.savefig('relation_entropy_random_forest.svg')
    plt.show()
    
    q = 10
    for _ in range(1):
        W = Wilson(G, q=q)
        F,roots = W.sample()
        print(W.s(), np.mean([len(Wilson(G, q=1/q).sample()[1]) for _ in range(reps)]) )

        to_networkx(G, F, roots)
        #plt.savefig('wilson_ring_of_cliques_5_10_beta_1E9.png')
        #print([x for x in F.nodes() if F.degree(x)==1])
        plt.show()