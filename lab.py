"""6.009 Lab 8: Graphs, Paths, Matrices."""

from abc import ABC, abstractmethod
# NO ADDITIONAL IMPORTS ALLOWED!


class Graph(ABC):
    """Interface for a mutable directed, weighted graph."""
    def __init__(self):
        # initialize the nodes, the neighbors and the edges existing in a graph 
        self.node = set()
        self.neighbor = {}
        self.edges = set()
        
#    @abstractmethod
    def add_node(self, node):
        """Add a node to the graph.

        Arguments:
            node (str): the node to add

        Raises:
            ValueError: if the node already exists.

        """
        if node in self.nodes():
            raise ValueError
        else:
            # if the node does not in the graph, add it to self.nodes and initialize its neighboring nodes to an empty set
            self.node.add(node)
            self.neighbor[node] = set()

#    @abstractmethod
    def add_edge(self, start, end, weight):
        """Add a directed edge to the graph.

        If the edge already exists, then set its weight to `weight`.

        Arguments:
            start (str): the node where the edge starts
            end (str): the node where the edge ends
            weight (int or float): the weight of the edge, assumed to be a nonnegative number

        Raises:
            LookupError: if either of these nodes doesn't exist

        """
        # if either of node not in the graph, raise LookupError
        if start not in self.nodes() or end not in self.nodes():
            raise LookupError
        else:
            # if the edge is already in the graph, then for each neighboring node of the start node, if it is equal to 
            # the end node, then first removing it with its old weight from the set of neighboring nodes of the start node
            if (start, end) in self.edges:
                for pair in self.neighbors(start):
                    if pair[0] == end:
                        self.neighbor[start].remove(pair)
                    break
                # then adding the end node with its new weight to the set of neighboring nodes of the start node
                self.neighbor[start].add((end, weight))
            else:
                # if the edge is not in the graph, then adding the edge to self.edges and add the end node to the set of 
                # neighboring nodes of the start node
                self.edges.add((start, end))
                self.neighbor[start].add((end, weight))
                
#    @abstractmethod
    def nodes(self):
        """Return the nodes in the graph.

        Returns:
            set: all of the nodes in the graph

        """
        # the getter of self.node
        return self.node.copy()
    
#    @abstractmethod
    def neighbors(self, node):
        """Return the neighbors of a node.

        Arguments:
            node (str): a node name

        Returns:
            set: all tuples (`neighbor`, `weight`) for which `node` has an
                 edge to `neighbor` with weight `weight`

        Raises:
            LookupError: if `node` is not in the graph

        """
        # the getter of self.neighbors
        if node not in self.neighbor:
            raise LookupError
        else:
            return self.neighbor[node].copy()

    @abstractmethod
    def get_path_length(self, start, end):
        """Return the length of the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            float or int: the length (sum of edge weights) of the path or
                          `None` if there is no such path.

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        

    @abstractmethod
    def get_path(self, start, end):
        """Return the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            list: nodes, starting with `start` and, ending with `end`, which
                  comprise the shortest path from `start` to `end` or `None`
                  if there is no such path

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        
    @abstractmethod
    def get_all_path_lengths(self):
        """Return lengths of shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to the length of the shortest path
                  from `u` to `v`

        """
        
    @abstractmethod
    def get_all_paths(self):
        """Return shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to a list of nodes (starting with
                  `u` and ending with `v`) which is a shortest path from `u`
                  to `v`

        """


class AdjacencyDictGraph(Graph):
    """A graph represented by an adjacency dictionary."""

    def __init__(self):
        """Create an empty graph."""
        Graph.__init__(self)
        # self.dist: the distance of any other nodes in the graph to a given start node 
        self.dist = {}
        # self.pred: the last node (value) that can travel to the current node (key)
        self.pred = {}
        
    def get_best_path(self, start):
        '''
        from this function, we can get the distance of any other nodes to a given start node
        '''
        # set all of nodes in the graph as unvisted 
        unvisited = self.nodes()
        # for each node in the graph, if the node is equal to the start node, set its distance to the start node to 0
        for n in self.nodes():
            if n == start:
                self.dist[n] = 0
            else:
                self.dist[n] = float('inf')
        
        # if there is any node that has not been visited, continue the loop
        while unvisited:
            # choose the node from unvisited that has the smallest distance from the given start node
            current_n = min(unvisited, key = lambda n: self.dist[n])
            # if the smallest distance to the start node is infinite, then stopping the loop
            if self.dist[current_n] == float('inf'):
                break
            # for each neighboring node of the chosen node, calculate the path length from start node to this neighboring node
            # based on the distance from the chosen node to the start node and the weight of edge between the chosen node and
            # the neighboring node
            for neighbor_pair in self.neighbors(current_n):
                neighbor_n = neighbor_pair[0]
                length = self.dist[current_n] + neighbor_pair[1]
                # if this alternative length is smaller than the current path length from start node to the neighboring node
                # storing in the self.dist, then update the value of the neighboring node in self.dist and set the node that
                # can travel to it in the shortest path to the chosen node
                if length < self.dist[neighbor_n]:
                    self.dist[neighbor_n] = length
                    self.pred[neighbor_n] = current_n
            # mark the chosen node as visited
            unvisited.remove(current_n)
            
    def set_shortest_path(self, start, end):
        shortest_path = []
        # set the end node of the shortest path to the node 'end'
        current_n = end
        # when the start node has not been reached, find the node that is connecting to the current node in the self.pred
        # and insert this node to the start of the current shortest_path list
        while current_n != start:
            if current_n in self.pred:
                shortest_path.insert(0, current_n)
                current_n = self.pred[current_n]
                
            else:
                return
        
        # add the node 'start' to the start of the shortest_path list
        if shortest_path != []:
            shortest_path.insert(0, current_n)
            
        # if no such a path, return None
        else:
            return
        
        return shortest_path
    
    def get_path_length(self, start, end):
        """Return the length of the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            float or int: the length (sum of edge weights) of the path or
                          `None` if there is no such path.

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        if start not in self.nodes() or end not in self.nodes():
            raise LookupError
        else:
            # get the distance of any other nodes in the graph to start
            self.get_best_path(start)
            # if the distance from start to end is infinite, then there is no path between two nodes, return None 
            if self.dist[end] == float('inf'):
                return
            else:
                return self.dist[end]
            
            
    def get_path(self, start, end):
        """Return the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            list: nodes, starting with `start` and, ending with `end`, which
                  comprise the shortest path from `start` to `end` or `None`
                  if there is no such path

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        if start not in self.nodes() or end not in self.nodes():
            raise LookupError
        
        # special case: if start is equal to end
        elif start == end:
            #print('yes')
            return [start]
        
        else:
            self.get_best_path(start)
            return self.set_shortest_path(start, end)
    
    def get_all_path_lengths(self):
        """Return lengths of shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to the length of the shortest path
                  from `u` to `v`

        """
        result = {}
        # for each start node in the graph, run the get_best_path method to get the distance from it to any other nodes in the graph
        for n1 in self.nodes():
            self.get_best_path(n1)
            # for each node that is not equal to the start node in the graph
            for n2 in self.nodes():
                # if no valid path between these two nodes, try another node 
                if self.dist[n2] == float('inf'):
                    continue
                else:
                    result[(n1, n2)] = self.dist[n2]
        
        return result
    
    def get_all_paths(self):
        """Return shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to a list of nodes (starting with
                  `u` and ending with `v`) which is a shortest path from `u`
                  to `v`

        """
        result = {}
        # for each start node in the graph
        for n1 in self.nodes():
            # initialize cond to True
            #cond = True
            # call get_best_path to get the distance from any other nodes to the current start node
            self.get_best_path(n1)
            for n2 in self.nodes():
                # if two nodes are equal, return [n1]
                if n1 == n2:
                    result[(n1, n2)] = [n1]
                # if no path between two nodes, try next node
                elif self.get_path_length(n1, n2) == None:
                    continue
                # else, repeat what is done in the get_path
                else:
                    shortest_path = self.set_shortest_path(n1, n2)
                    if shortest_path != None:
                        result[(n1, n2)] = shortest_path
        return result

class AdjacencyMatrixGraph(Graph):
    """A graph represented by an adjacency matrix."""

    def __init__(self):
        """Create an empty graph."""
        Graph.__init__(self)
        self.dist = []
        self.pred = []
    
    # convert the name of a node to the index and convert the index of a node to its name
    def convert(self, node):
        # sort all of nodes in the graph
        node_list = sorted(list(self.nodes()))
        # if the input is an index of a node, convert it to the name of this node 
        if type(node) == int:
            return node_list[node]
        # if the input is a string, convert it to its index in the node_list
        elif node in node_list:
            return node_list.index(node)
        # if the node is a string that is not in the graph, return None
        else:
            return
        
    def get_best_path(self, start):
        # build up the matrix first; assuming that the distance between any pair of nodes is infinite at this time
        for i in range(len(self.nodes())):
            row = []
            for j in range(len(self.nodes())):
                row.append(float('inf'))
            self.dist.append(row)
        
        # Initialize the self.pred
        for i in range(len(self.nodes())):
            row = []
            for j in range(len(self.nodes())):
                row.append(None)
            self.pred.append(row)
        
        unvisited = set(range(len(self.nodes())))
        start_idx = self.convert(start)
            
        self.dist[start_idx][start_idx] = 0
        
        # using the similar method as dictionary graph representation
        while unvisited:
            # choose the node from unvisited that has the smallest distance from the given start node
            current_n = min(unvisited, key = lambda n: self.dist[start_idx][n])
            if self.dist[start_idx][current_n] == float('inf'):
                break
            
            # for each neighboring node of the chosen node, calculate the path length from start node to this neighboring node
            # based on the distance from the chosen node to the start node and the weight of edge between the chosen node and
            # the neighboring node
            for neighbor_pair in self.neighbors(self.convert(current_n)):
                neighbor_n = neighbor_pair[0]
                # if this alternative length is smaller than the current path length from start node to the neighboring node
                # storing in the self.dist, then update the value of the neighboring node in self.dist and set the node that
                # can travel to it in the shortest path to the chosen node
                length = self.dist[start_idx][current_n] + neighbor_pair[1]
                neighbor_idx = self.convert(neighbor_n)
                if length < self.dist[start_idx][neighbor_idx]:
                    self.dist[start_idx][neighbor_idx] = length
                    self.pred[start_idx][neighbor_idx] = current_n
            # mark the chosen node as visited
            unvisited.remove(current_n)
    
    def set_shortest_path(self, start, end):
        shortest_path = []
        # convert the start and end node to its corresponding index
        start_idx = self.convert(start)
        end_idx = self.convert(end)
        # set the end node of the shortest path to the node 'end'
        current_n = end_idx
        # when the start node has not been reached, find the node (in index mode) that is connecting to the current node in the self.pred
        # and insert this node (in name mode) to the start of the current shortest_path list
        while current_n != start_idx:
            if self.pred[start_idx][current_n] != None:
                current_n_name = self.convert(current_n)
                shortest_path.insert(0, current_n_name)
                current_n = self.pred[start_idx][current_n]
                
            else:
                return
        
        # if the path is valid, add the start node to the beginning of the list
        if shortest_path != []:
            current_n_name = self.convert(current_n)
            shortest_path.insert(0, current_n_name)
        else:
            return
        
        return shortest_path
    
    def get_path_length(self, start, end):
        """Return the length of the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            float or int: the length (sum of edge weights) of the path or
                          `None` if there is no such path.

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        if start not in self.nodes() or end not in self.nodes():
            raise LookupError
        else:
            if start == end:
                return 0
            # call get_best_path to get the distance from start node to any other nodes in the graph
            self.get_best_path(start)
            # convert start and end node to its index
            start_idx = self.convert(start)
            end_idx = self.convert(end)
            # if the path is invalid (distance between two nodes is infinite), return None
            if self.dist[start_idx][end_idx] == float('inf'):
                return
            else:
                return self.dist[start_idx][end_idx]
            
            
    def get_path(self, start, end):
        """Return the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            list: nodes, starting with `start` and, ending with `end`, which
                  comprise the shortest path from `start` to `end` or `None`
                  if there is no such path

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        # two special cases
        if start not in self.nodes() or end not in self.nodes():
            raise LookupError
            
        elif start == end:
            return [start]
        
        # normal case
        else:
            # call get_best_path to get the information needed by the function set_shortest_path
            self.get_best_path(start)
            return self.set_shortest_path(start, end)
        
    
    def get_all_path_lengths(self):
        """Return lengths of shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to the length of the shortest path
                  from `u` to `v`

        """
        result = {}
        # for each start node in the graph, call get_best_path to get the distance from this node to any other nodes 
        for n1 in self.nodes():
            self.get_best_path(n1)
            for n2 in self.nodes():
                start_idx = self.convert(n1)
                end_idx = self.convert(n2)
                # if the distance between two nods is infinite, try next node
                if self.dist[start_idx][end_idx] == float('inf'):
                    continue
                else:
                    result[(n1,n2)] = self.dist[start_idx][end_idx]
        
        return result
    
    def get_all_paths(self):
        """Return shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to a list of nodes (starting with
                  `u` and ending with `v`) which is a shortest path from `u`
                  to `v`

        """
        result = {}
        # for each start node in the graph, call get_best_path to get the distance from this node to any other nodes
        for n1 in self.nodes():
            self.get_best_path(n1)
            # for each node that is not equal to the current start node
            for n2 in self.nodes():
                if n1 == n2:
                    result[(n1, n2)] = [n1]
                # if no valid path between two nodes, try next noede
                elif self.get_path_length(n1, n2) == None:
                    continue
                
                # call set_shortest_path to build up the shortest path list
                else:
                    shortest_path = self.set_shortest_path(n1, n2)
                    if shortest_path != None:
                        result[(n1, n2)] = shortest_path
        return result
    
class GraphFactory:
    """Factory for creating instances of `Graph`."""

    def __init__(self, cutoff=0.5):
        """Create a new factory that creates instances of `Graph`.

        Arguments:
            cutoff (float): the maximum density (as defined in the lab handout)
                            for which the an `AdjacencyDictGraph` should be
                            instantiated instead of an `AdjacencyMatrixGraph`

        """
        self.cutoff = cutoff

    def from_edges_and_nodes(self, weighted_edges, nodes):
        """Create a new graph instance.

        Arguments:
            weighted_edges (list): the edges in the graph given as
                                   (start, end, weight) tuples
            nodes (list): nodes in the graph

        Returns:
            Graph: a graph containing the given edges

        """
        # First calculate the ratio of existing weighted edges to the total possible edges for all nodes in the graph
        ratio = len(weighted_edges)/(len(nodes)*(len(nodes) - 1))
        # choose the graph representation according to the ratio and given cutoff value
        if ratio <= self.cutoff:
            new_graph = AdjacencyDictGraph()
        else:
            new_graph = AdjacencyMatrixGraph()
        
        # add each node in the given nodes to the graph
        for node in nodes:
            new_graph.add_node(node)
        
        # add each weighted edge into the graph
        for edge in weighted_edges:
            new_graph.add_edge(edge[0], edge[1], edge[2])
        
        return new_graph

def get_most_central_node(graph):
    """Return the most central node in the graph.

    "Most central" is defined as having the shortest average round trip to any
    other node.

    Arguments:
        graph (Graph): a graph with at least one node from which round trips
                       to all other nodes are possible

    Returns:
        node (str): the most central node in the graph; round trips to all
                    other nodes must be possible from this node

    """
    
    # First get all of valid path length in the graph
    result = graph.get_all_path_lengths()
    # initialize the current shortest average length to infinity
    shortest_length = float('inf')
    central_n = None
    # for each node in the graph
    for n1 in graph.nodes():
        # initialize cond as True and initialize the total length of path through which this node connects to any other node in th graph
        cond = True
        total_length = 0
        # for any other nodes in the graph
        for n2 in graph.nodes():
            if n1 == n2:
                continue
            # if there is no path between two nodes, then n1 cannot be the most central node; in this case, set cond to False
            # and try the next n1 in the graph
            if (n1, n2) not in result:
                cond = False
                break
            # else, update the total length for n1
            else:
                total_length += result[(n1, n2)]
        
        # if cond is True, which means the current n1 can connect to any other nodes in the graph, then calculating the average
        # path length and compare it with the current shortest_length
        if cond:
            average_length = total_length/(len(graph.nodes()) - 1)
            if average_length < shortest_length:
                shortest_length = average_length
                central_n = n1
                
    return central_n

if __name__ == "__main__":
    # You can place code (like custom test cases) here that will only be
    # executed when running this file from the terminal.
    graph_fact = GraphFactory()
    weighted_edges = [('a', 'b', 2), ('b','c', 5),('c','a', 4)]
    nodes = ['a','b','c']
    graph = graph_fact.from_edges_and_nodes(weighted_edges, nodes)
    result = get_most_central_node(graph)
    print(result)
    
