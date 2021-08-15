# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Set
import itertools
import networkx as nx
from networkx.drawing.nx_pylab import draw
from networkx.algorithms.moral import moral_graph


class JunctionTree:
    '''class containing a junction tree'''
    def __init__(self, clusters, cluster_tuples, complete_graph):
        self.clusters = clusters
        self.cluster_tuples = cluster_tuples
        self.complete_graph = complete_graph
        self.min_span_tree = nx.minimum_spanning_tree(self.complete_graph)

    def show_tree(self):
        '''shows the tree'''
        #TODO should utilize ___str___
        print(f"Clusters as lists of edges: {self.clusters}")
        print(f"Clusters as lists of nodes: {self.cluster_tuples}")
        print(self.complete_graph)
        print(
            f"Edges between clusters (pairs of clusters): {self.min_span_tree.edges()}"
        )


class JunctionTreeFromBayesianGraph:
    """
    Creates a Junction Tree from possibly cycled directed bayesian graph or a factor graph.
    
    Consider a following product of factors

    .. math::

        \phi(1,2,4)\phi(2,3,4)\phi(4,5)\phi(4,6,7)\phi(1,7)

    Based on the above factors we create the following list of factors as an input to the algrithm creating a Junction Tree. 

    C = [(1, 2, 4), (2, 3, 4), (4, 5), (4, 6, 7), (1, 7)]

    This representation requires usage of the alternative constructor provided by the static method from_factors.

    Running the Junction Tree algorithm over the above factor representation of the graph yields

    .. todo:: Add sample output

    Default constructor takes as input an undirected moral graph. 

    :param graph: the undirected moral graph created by one of the classmethods 
    :type graph: nx.Graph

    """
    def __init__(self, graph : nx.Graph):

        self.H = graph

        print(list(self.H.nodes))
        print(list(self.H.edges))

    #  @classmethod
    #  def from_bayesian_graph(cls, C: Dict[int, List]) -> JunctionTreeFromBayesianGraph:
    #      """Creates a moral graph from a dictionary representing the directed bayesian graph. In this representation each node has to be exactly one time as head.
    #
    #      :param C: input dictonary with heads ad keys and tails as values
    #      :type C: Dict[int, List]
    #      :rtype: JunctionTreeFromBayesianGraph
    #      """
    #      edges = []
    #      nodes = []
    #      for head in C:
    #          if head not in nodes:
    #              nodes.append(head)
    #          for tail in C[head]:
    #              if (tail, head) not in edges and tail != '':
    #                  edges.append((tail, head))
    #      return cls(nodes, edges)

    @classmethod
    def from_c_dict(cls, C : Dict[int, List]) -> JunctionTreeFromBayesianGraph:
        """Creates a moral graph from a dictionary representing the directed bayesian graph. In this representation each node has to be exactly one time as head.

        :param C: input dictonary with heads ad keys and tails as values
        :type C: Dict[int, List]
        :return: an object of JunctionTreeFromBayesianGraph type
        :rtype: JunctionTreeFromBayesianGraph
        """

        # create cliques
        cliques = []
        nodes = set()

        for head in C.keys():

            clique = [head]

            clique.extend(C[head])

            print(f"clique {clique}")

            nodes.add(head)

            nodes.union(tuple(C[head]))

            cliques.append(clique)

        graph = nx.Graph()

        # add edges to graph

        for clique in cliques:

            print(clique)

            edges_per_clique = itertools.combinations(clique,2)

            graph.add_edges_from(edges_per_clique)

        graph.add_nodes_from(list(nodes))

        return cls(graph)

    @classmethod
    def from_factors(cls, factors : List[Tuple]) -> JunctionTreeFromBayesianGraph:
        """Creates cliques for each factor - a moral graph.

        :param factors: a list of tuples, each tuple contains factor variables labels (ints)
        :type factors: List[Tuple]
        :return: an object of JunctionTreeFromBayesianGraph type
        :rtype: JunctionTreeFromBayesianGraph
        """

        graph = nx.Graph()

        nodes = set()

        nodes.union(*factors)

        # remove spurious factors

        factors_clean = []

        for i in range(len(factors)):
            factor_is_spurious = False
            print(set(list(factors[i])))
            for j in range(i):
                if set(list(factors[i])).issubset(set(list(factors[j]))):
                    factor_is_spurious = True

            for j in range(i+1,len(factors)):
                if set(list(factors[i])).issubset(set(list(factors[j]))):
                    factor_is_spurious = True

            print(f"sf {factor_is_spurious}, factor {factors[i]}")

            if not factor_is_spurious:

                factors_clean.append(factors[i])

        for clique in factors_clean:

            print(f"factor {clique}")

            edges_per_clique = itertools.combinations(clique,2)

            graph.add_edges_from(edges_per_clique)



        graph.add_nodes_from(list(nodes))

        return cls(graph)

    """Helper functions for the elimination ordering"""

    def _get_edges_set(self) -> Set:
        """Converts list of edges into set of edges.

        :rtype: Set
        """
        return set(self.H.edges())

    def _check_edge_repeat(self, edge: Tuple, edges: Set) -> bool:
        """Check if given edge is already in the edges set.

        :param edge: 2-tuple, edge
        :type edge: Tuple
        :param edges: set of edges
        :type edges: Set
        :rtype: bool
        """
        return edge not in edges and (edge[1], edge[0]) not in edges

    def _minfillcost(self, edges: Set, naighbour_nodes: List) -> int:
        """Computes minfillcost for the variable elimination heuristics.

        :param edges: set of edges
        :type edges: Set
        :param neighbour_nodes: list of neighbour nodes 
        :type neighbour_nodes: List
        :rtype: int
        """

        elimination_clique = itertools.combinations(naighbour_nodes, 2)
        cost = sum([(self._check_edge_repeat(edge, edges))
                    for edge in elimination_clique])
        return cost

    def _find_node(self, H_copy: nx.Graph) -> int:
        """Finds node with minimal fillcost.

        :param H_copy: the graph
        :type H_copy: nx.Graph
        :rtype: int
        """
        edges = self._get_edges_set()
        costs = [(v, self._minfillcost(edges, H_copy.neighbors(v)))
                 for v in list(H_copy)]
        print(f"costs {costs}")
        costs.sort(key=lambda e: e[1])
        node = costs[0][0]
        return node

    def create_elimination_ordering(self) -> List[int]:
        """
        Find repeatedly vertex with minimal elimination cost, create elimination ordering.

        :rtype: List[int]
        """
        elimination_ordering = []
        H_copy = self.H.copy()
        while list(H_copy):
            node = self._find_node(H_copy)
            elimination_ordering.append(node)
            H_copy.remove_node(node)
        return elimination_ordering

    def _check_subset(self, cluster_include_to : Set, cluster_included : Set) -> bool:

        return cluster_included.issubset(cluster_include_to)

    def _cluster_already_included(self, clusters : List, cluster_included : Set) -> bool:

        included = False

        clusters = self._turn_clusters_into_sets(clusters)

        print(f"cluster sets {clusters}")
        print(f"clusters included {cluster_included}")

        for cluster in clusters:

            included = self._check_subset(cluster, cluster_included)

        return included

    def find_clusters(self, elimination_ordering: List[int]) -> List[List]:
        """Find clusters in a moral graph using elimination ordering.

        :param elimination_ordering: the elimination ordering as returned by the method create_elimination_ordering
        :type elimination_ordering: List[int]
        :rtype: List[List]
        """
        clusters = []
        for node in elimination_ordering:
            neighbor_nodes = list(self.H.neighbors(node))
            neighbor_nodes.append(node)
            elimination_clique = itertools.combinations(neighbor_nodes, 2)
            print(f"elimination clique {elimination_clique}")
            elimination_clique_lst = list(elimination_clique)
            print(f"elmination clique lst {elimination_clique_lst}")
            print(f"clusters {clusters}")
            print(f"elimination clique {elimination_clique_lst}")
            if not self._cluster_already_included(clusters, set(itertools.chain(*elimination_clique_lst))):
                clusters.append(elimination_clique_lst)
            edges = self._get_edges_set()
            for edge in elimination_clique_lst:
                if self._check_edge_repeat(edge, edges):
                    self.H.add_edge(*edge)
            if len(edges) == len(elimination_clique_lst):
                break
            self.H.remove_node(node)
        return clusters

    def _turn_clusters_into_sets(self, clusters: List[List]) -> List[Set]:
        """Converts clusters to sets thus eliminating duplicate nodes.

        :param clusters: clusters as a list of lists
        :type clusters: List[List]
        :rtype: List[Set]
        """
        cluster_sets = []
        for cluster in clusters:
            cluster_sets.append(set(itertools.chain(*cluster)))
        return cluster_sets

    def _turn_clusters_into_tuples(self,
                                   cluster_sets: List[set]) -> List[Tuple]:
        """Converts clusters represented as sets into tuples.

        :param cluster_sets: list of cluster sets
        :type cluster_sets: List[set]
        :rtype: List[Tuple]
        """
        cluster_tuples = []
        for cluster in cluster_sets:
            cluster_tuples.append(tuple(cluster))
        return cluster_tuples

    def remove_clusters_duplicates(self, clusters: List[List]) -> List[Tuple]:
        """Removes duplicates from clusters, returns cluster tuples.

        :param clusters: clusters
        :type clusters: List[List]
        :rtype: List[Tuple]
        """
        cluster_sets = self._turn_clusters_into_sets(clusters)
        cluster_tuples = self._turn_clusters_into_tuples(cluster_sets)
        return cluster_tuples

    def run_max_span_tree_algorithm(self,
                                    cluster_tuples: List[Tuple]) -> nx.Graph():
        """Using clusters tuples without duplicates, calculates complete graph using minimum spanning tree algorithm.

        :param cluster_tuples: clusters
        :type cluster_tuples: List[Tuple]
        :rtype: nx.Graph()
        """
        complete_graph = nx.Graph()
        edges = list(itertools.combinations(cluster_tuples, 2))
        weights = list(
            map(lambda x: len(set(x[0]).intersection(set(x[1]))), edges))
        for edge, weight in zip(edges, weights):
            complete_graph.add_edge(*edge, weight=-weight)
        return complete_graph

    def compute(self) -> JunctionTree:
        """Executes the complete algorithm forming the JunctionTree object at the end. This is the main API function.

        :rtype: JunctionTree
        """
        elimination_ordering = self.create_elimination_ordering()
        print(f"elimin ord {elimination_ordering}")
        return self.compute_given_elimination_ordering(elimination_ordering) 

    def compute_given_elimination_ordering(self, elimination_ordering : List) -> JunctionTree:
        """Executes the complete algorithm given predefined elimination ordering forming the JunctionTree object at the end. 
        This is the main API function.

        :rtype: JunctionTree
        """

        clusters = self.find_clusters(elimination_ordering)
        cluster_tuples = self.remove_clusters_duplicates(clusters)
        complete_graph = self.run_max_span_tree_algorithm(cluster_tuples)
        return JunctionTree(clusters, cluster_tuples, complete_graph)
if __name__ == "__main__":

    # example of executing the JunctionTree computation
    # input graph

    F = [(1,2,4), (2,3,4), (4,5), (4,6,7), (1,7)]

    #  F = [(1,3),(2,3)]

    # object creation using static method
    jt = JunctionTreeFromBayesianGraph.from_factors(F)

    # running the algorithm
    junction_tree = jt.compute_given_elimination_ordering([7,6,5,4,3,2,1])
    #  junction_tree = jt.compute()
    # showing results
    junction_tree.show_tree()
