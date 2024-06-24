"""
Thie file contains the Graph and WeightedGraph classes, as well as the
required function to load a weighted recipe review graph.

Authors: Yiping Chen, Alan Su, Lily Phan, Defne Eris
Professor: Sadia Sharmin
Course: CSC111, Introduction to Computer Science
Date: April 2024
"""

# Import statements
from __future__ import annotations
import csv
from datetime import date
from ast import literal_eval
from typing import Union, Optional
from random import choices, random
import json
import networkx as nx
from tqdm import tqdm

from metadata import Interaction


class _Vertex:
    """A vertex in a recipe review graph, used to represent a user or a recipe.

    Instance Attributes:
        - id: The id of the recipe or user corresponding to the vertex
        - is_recipe: Whether or not this instance of a vertex represents a recipe, otherwise user
        - neighbours: The vertices that are adjacent to this vertex.

    Representation Invariants:
        - all(self in u.neighbours for u in self.neighbours)
    """
    id: int
    is_recipe: bool
    neighbours: set[_Vertex]

    def __init__(self, vertex_id: int, is_recipe: bool) -> None:
        """Initialize a new vertex with the given id and whether it is a recipe or user.
        This vertex is initialized with no neighbours.
        """
        self.id = vertex_id
        self.is_recipe = is_recipe
        self.neighbours = set()

    def degree(self) -> int:
        """Return the degree of this vertex."""
        return len(self.neighbours)


class Graph:
    """A graph used to represent a recipe review network.

    Private Instance Attributes:
        - _vertices: A dictionary of the vertices contained in this graph, mapping their id to the _Vertex object
    """
    _vertices: dict[int, _Vertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

    def add_vertex(self, item: int, is_recipe: bool) -> None:
        """Add a vertex with the given item and kind to this graph.

        The new vertex is not adjacent to any other vertices.
        Do nothing if the given item is already in this graph.
        """
        if item not in self._vertices:
            self._vertices[item] = _Vertex(item, is_recipe)

    def add_edge(self, item1: int, item2: int) -> None:
        """Add an edge between the two vertices with the given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            v1.neighbours.add(v2)
            v2.neighbours.add(v1)
        else:
            raise ValueError

    def adjacent(self, item1: int, item2: int) -> bool:
        """Return whether item1 and item2 are adjacent vertices in this graph.

        Return False if item1 or item2 do not appear as vertices in this graph.
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            return any(v2.id == item2 for v2 in v1.neighbours)
        else:
            return False

    def get_neighbours(self, item: int) -> set:
        """Return a set of the neighbours of the given item.

        Note that the ids are returned, not the _Vertex objects themselves.

        Raise a ValueError if item does not appear as a vertex in this graph.
        """
        if item in self._vertices:
            v = self._vertices[item]
            return {neighbour.id for neighbour in v.neighbours}
        else:
            raise ValueError

    def get_all_vertices(self, is_recipe: Optional[bool] = None) -> set:
        """Return a set of all vertex items in this graph. If is_recipe is given, then only return vertices
        corresponding to the same type. If is_recipe is None, then all vertices are returned.

        Note that the ids are returned, not the _Vertex objects themselves
        """
        if is_recipe is not None:
            return {v.id for v in self._vertices.values() if v.is_recipe == is_recipe}
        else:
            return set(self._vertices.keys())

    def to_networkx(self, max_vertices: int = 5000) -> nx.Graph:
        """Convert this graph into a networkx Graph.

        max_vertices specifies the maximum number of vertices that can appear in the graph.
        (This is necessary to limit the visualization output for large graphs.)
        """
        graph_nx = nx.Graph()
        for v in self._vertices.values():
            graph_nx.add_node(v.id, kind=v.is_recipe)

            for u in v.neighbours:
                if graph_nx.number_of_nodes() < max_vertices:
                    graph_nx.add_node(u.id, kind=u.is_recipe)

                if u.id in graph_nx.nodes:
                    graph_nx.add_edge(v.id, u.id)

            if graph_nx.number_of_nodes() >= max_vertices:
                break

        return graph_nx


class _WeightedVertex(_Vertex):
    """A vertex in a weighted recipe review graph, used to represent a user or a recipe.

    Instance Attributes:
        - id: The id of the recipe or user stored in this vertex
        - is_recipe: Whether or not the vertex is a recipe, or user otherwise
        - neighbours: The vertices that are adjacent to this vertex, and their corresponding review interactions
        - similarity_scores: A dictionary mapping a vertex to a dictionary storing its similarity to every
            vertex, including itself
        - metadata: A dictionary only present in recipe vertices which maps recipe metadata such as name, description,
            and ingredients to their data

    Representation Invariants:
        - all(self in u.neighbours for u in self.neighbours)
        - not self.is_recipe and not self.metadata or self.is_recipe and self.metadata
    """
    id: int
    is_recipe: bool
    neighbours: dict[_WeightedVertex, Interaction]
    similarity_scores: dict[bool, dict[str, dict[_WeightedVertex, float]]]
    metadata: Optional[dict[str, Union[int, str, list[str], list[int], date]]]
    sentimental: bool

    def __init__(self, vertex_id: int, is_recipe: bool, metadata: dict = None) -> None:
        """Initialize a new vertex with the given item and kind. Similarity scores for Simrank and Simrank++ are by
        default 1.0 for the same vertex.

        This vertex is initialized with no neighbours.
        """
        super().__init__(vertex_id, is_recipe)
        self.neighbours = {}
        default_similarity_scores = {'simrank': {self: 1.0}, 'simrank_pplus': {self: 1.0}}
        self.similarity_scores = {True: default_similarity_scores.copy(), False: default_similarity_scores.copy()}
        self.metadata = metadata

    def jaccard_sim(self, other: _WeightedVertex, sentimental: bool = False) -> float:
        """Returns the Jaccard similarity score between this vertex and the other.
        """
        min_edges, max_edges = 0, 0
        for v in set(self.neighbours).union(set(other.neighbours)):
            self_edge = 0.0 if v not in self.neighbours else self.neighbours[v].get_weight(sentimental)
            other_edge = 0.0 if v not in other.neighbours else other.neighbours[v].get_weight(sentimental)
            min_edges += min(self_edge, other_edge)
            max_edges += max(self_edge, other_edge)
        if max_edges == 0:
            return 0
        return min(min_edges / max_edges, 1.0)

    def overlap_sim(self, other: _WeightedVertex, sentimental: bool = False) -> float:
        """Returns the Overlap similarity score between this vertex and the other.
        """
        min_edges, cur_edges, other_edges = 0, 0, 0
        for v in set(self.neighbours).union(set(other.neighbours)):
            cur_rating = 0.0 if v not in self.neighbours else self.neighbours[v].get_weight(sentimental)
            other_rating = 0.0 if v not in other.neighbours else other.neighbours[v].get_weight(sentimental)
            min_edges += min(cur_rating, other_rating)
            cur_edges += cur_rating
            other_edges += other_rating
        min_sum = min(cur_edges, other_edges)
        if min_sum == 0:
            return 0
        return min(min_edges / min_sum, 1.0)

    def cosine_sim(self, other: _WeightedVertex, sentimental: bool = False) -> float:
        """Returns the Cosine similarity score between this vertex and the other.
        """
        total_edges, cur_edges, other_edges = 0, 0, 0
        for v in set(self.neighbours).union(set(other.neighbours)):
            cur_rating = 0.0 if v not in self.neighbours else self.neighbours[v].get_weight(sentimental)
            other_rating = 0.0 if v not in other.neighbours else other.neighbours[v].get_weight(sentimental)
            total_edges += cur_rating * other_rating
            cur_edges += cur_rating ** 2
            other_edges += other_rating ** 2
        norm_edges = cur_edges ** 0.5 * other_edges ** 0.5
        if norm_edges == 0:
            return 0
        return min(total_edges / norm_edges, 1.0)

    def tanimoto_sim(self, other: _WeightedVertex, sentimental: bool = False) -> float:
        """Returns the Tanimoto similarity score between this vertex and the other.
        """
        total_edges, cur_edges, other_edges = 0, 0, 0
        for v in set(self.neighbours).union(set(other.neighbours)):
            cur_rating = 0.0 if v not in self.neighbours else self.neighbours[v].get_weight(sentimental)
            other_rating = 0.0 if v not in other.neighbours else other.neighbours[v].get_weight(sentimental)
            total_edges += cur_rating * other_rating
            cur_edges += cur_rating ** 2
            other_edges += other_rating ** 2
        norm_edges = cur_edges ** 0.5 + other_edges ** 0.5 - total_edges
        if norm_edges == 0:
            return 0
        return min(total_edges / norm_edges, 1.0)

    def simrank_sim(self, other: _WeightedVertex, sentimental: bool = False) -> float:
        """Returns the Simrank similarity score betweeen this vertex and the other. Either the precomputed similarities
        must have been loaded or the Simrank must have been calculated.
        """
        return self.similarity_scores[sentimental]['simrank'][other]

    def simrank_pplus_sim(self, other: _WeightedVertex, sentimental: bool = False) -> float:
        """Returns the Simrank++ similarity score betweeen this vertex and the other. Either the precomputed
        similarities must have been loaded or the Simrank++ must have been calculated.
        """
        return self.similarity_scores[sentimental]['simrank_pplus'][other]

    def pagerank_sim(self, other: _WeightedVertex, sentimental: bool = False) -> float:
        """Returns the Personalized Pagerank similarity socre between this vertex and the other. Either the precomputed
        similarities must have been loaded or the Personalized Pagerank must have been calculated.
        """
        return self.similarity_scores[sentimental]['pagerank'][other]

    def calculate_simrank(self, other: _WeightedVertex, decay_factor: float, pplus: bool,
                          sentimental: bool = False) -> float:
        """Calculates the Simrank value between ths vertex and the other vertex, depending on the decay_factor
        argument. If pplus is True, then the Simrank++ value is calculated instead.

        Preconditions:
            - 0.0 <= decay_factor <= 1.0
        """
        if self == other:
            return 1.0
        if len(self.neighbours) == 0 or len(other.neighbours) == 0:
            return 0.0
        simrank_sum = 0.0
        for u in self.neighbours:
            for v in other.neighbours:
                vertex_simrank = u.similarity_scores[sentimental]['simrank'] if not pplus else \
                    u.similarity_scores[sentimental]['simrank_pplus']
                self_weight, other_weight = self.neighbours[u].get_weight(sentimental), other.neighbours[
                    v].get_weight(sentimental)
                if not sentimental:
                    self_weight /= 5
                    other_weight /= 5
                else:
                    self_weight = (self_weight + 1) / 2
                    other_weight = (other_weight + 1) / 2
                if v in vertex_simrank:
                    simrank_sum += vertex_simrank[v] * self_weight * other_weight
        score = decay_factor / (len(self.neighbours) * (len(other.neighbours))) * simrank_sum
        if pplus:
            size = len(set(self.neighbours.keys()).intersection(set(other.neighbours.keys())))
            return sum(1 / (2 ** i) for i in range(1, size + 1)) * score
        else:
            return score

    def random_walk(self, vertices: set[_WeightedVertex], damping_factor: float,
                    error_threshold: float, sentimental: bool = False) -> dict[_WeightedVertex, float]:
        """Performs a random walk starting at self depending on the set of vertices. The random walk is personalized,
        such that upon teleportation, the walk will always return to this self vertex. The error_threshold determines
        the number of iterations such that the floats returned are within the error threshold of their convergent
        value. Returns a dictinoary mapping the other vertices to their corresponding similarity to the self vertex,
        depending on the damping factor.

        Preconditions:
            - 0.0 <= damping_factor <= 1.0
            - error_threshold > 0.0
        """
        visited_cnt = {v: 0 for v in vertices}
        cur_vertex = self
        visited_cnt[cur_vertex] += 1
        num_iterations = int(1 / error_threshold)
        for _iteration in range(num_iterations):
            if random() < damping_factor:
                neighbours = [(v, interaction.get_weight(sentimental)) for v, interaction in
                              cur_vertex.neighbours.items()]
                neighbour_vertices, neighbour_weights = [x[0] for x in neighbours], [x[1] for x in neighbours]
                cur_vertex = choices(neighbour_vertices, weights=neighbour_weights, k=1)[0]
            else:
                cur_vertex = self
            visited_cnt[cur_vertex] += 1
        total_cnt = sum(x for x in visited_cnt.values())
        return {v: (cnt / total_cnt) for v, cnt in visited_cnt.items()}


class WeightedGraph(Graph):
    """A weighted graph used to represent a recipe review network that keeps track of review scores.

    Private Instance Attributes:
        - _vertices: A dictionary of the vertices contained in this graph, mapping their id to the _WeightedVertex
            object
    """
    _vertices: dict[int, _WeightedVertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

        # This call isn't necessary, except to satisfy PythonTA.
        Graph.__init__(self)

    def add_vertex(self, item: int, is_recipe: bool,
                   metadata: Optional[dict[str, Union[int, str, list[str], list[int], date]]] = None) -> None:
        """Add a vertex with the given item and whether or not it is a recipe to the graph. If the vertex is
        a recipe, then it should include the recipe metadata.

        The new vertex is not adjacent to any other vertices.
        Do nothing if the given item is already in this graph.

        Preconditions:
            - is_recipe and metadata or not is_recipe and not metadata
        """
        if item not in self._vertices:
            self._vertices[item] = _WeightedVertex(item, is_recipe, metadata)

    def add_edge(self, item1: int, item2: int, interaction: Interaction = Interaction(1)) -> None:
        """Add an edge between the two vertices with the given items in this graph,
        with the given weight. If no interaction (representing the user's review and rating) is given, the default to
        a one weight edge.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            # Add the new edge
            v1.neighbours[v2] = interaction
            v2.neighbours[v1] = interaction
        else:
            # We didn't find an existing vertex for both items.
            raise ValueError

    def get_weight(self, item1: int, item2: int) -> Union[int, float]:
        """Return the weight of the edge between the given items, frmo the Interaction class' rating.

        Return 0 if item1 and item2 are not adjacent.

        Preconditions:
            - item1 and item2 are vertices in this graph
        """
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]
        return v1.neighbours.get(v2, Interaction(rating=0)).rating

    def to_networkx(self, max_vertices: int = 5000) -> nx.Graph:
        """Convert this graph into a networkx Graph.

        max_vertices specifies the maximum number of vertices that can appear in the graph.
        (This is necessary to limit the visualization output for large graphs.)

        Note that this method is provided for you, and you shouldn't change it.
        """
        graph_nx = nx.Graph()
        for v in self._vertices.values():
            graph_nx.add_node(v.id, kind=v.is_recipe)

            for u in v.neighbours.keys():
                if graph_nx.number_of_nodes() < max_vertices:
                    graph_nx.add_node(u.id, kind=u.is_recipe)

                if u.id in graph_nx.nodes:
                    graph_nx.add_edge(v.id, u.id, weight=v.neighbours[u].rating)

            if graph_nx.number_of_nodes() >= max_vertices:
                break

        return graph_nx

    def get_all_recipe_names(self) -> set[str]:
        """Returns a set of all recipe names for each vertex in the graph.
        """
        return {v.metadata['name'] for v in self._vertices.values() if v.is_recipe}

    def densest_subgraph(self, subgraph_size: int = 500, file_path: str = 'data/vertices_small.txt') -> None:
        """Removes the vertex with the least number of neighbours until self becomes a graph with a subgraph_size
        number of vertices. Then, saves the ids of the vertices in the new graph into a text file depending on the
        file_path.

        Preconditions:
            - 0 <= subgraph_size <= len(self._vertices)
        """
        total_size = len(self._vertices)
        for _i in tqdm(range(total_size - subgraph_size)):
            min_vertex_id, min_vertex = min([(v.id, v) for v in self._vertices.values()],
                                            key=lambda x: len(x[1].neighbours))
            for neighbour in min_vertex.neighbours:
                if neighbour != min_vertex:
                    del neighbour.neighbours[min_vertex]
            del self._vertices[min_vertex_id]

        with open(file_path, 'w', encoding='utf-8') as file:
            for vertex in self._vertices:
                file.write(f"{vertex}\n")

    def generate_subgraph_dataset(self, file_paths: dict[str, str] = None) -> None:
        """Reads a set of vertices and filters a csv of recipes and interactions to only include those vertices.
        """
        default_file_paths = {
            'new_vertices': 'data/vertices_small.txt',
            'interactions': 'data/interactions.csv',
            'recipes': 'data/interactions.csv',
            'new_interactions': 'data/new_interactions.csv',
            'new_recipes': 'data/new_recipes.csv'
        }
        if file_paths is None:
            file_paths = default_file_paths
        for file_path, path in default_file_paths.items():
            if file_path not in file_paths:
                file_paths[file_path] = path
        with open(file_paths['new_vertices'], 'r', encoding='utf-8') as file:
            allowed_ids = {int(x) for x in file.readlines()}
        with open(file_paths['recipes'], 'r', encoding='utf-8') as recipes_file, open(file_paths['new_recipes'], 'w',
                                                                                      encoding='utf-8') as recipes_out:
            reader = csv.reader(recipes_file)
            writer = csv.writer(recipes_out)
            next(reader, None)
            writer.writerow(
                ['name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps',
                 'description', 'ingredients', 'n_ingredients'])
            for row in reader:
                if int(row[1]) in allowed_ids:
                    writer.writerow(row)
        with open(file_paths['interactions'], 'r', encoding='utf-8') as interactions_file, open(
                file_paths['new_interactions'],
                'w', encoding='utf-8') as interactions_out:
            reader = csv.reader(interactions_file)
            writer = csv.writer(interactions_out)
            next(reader, None)
            writer.writerow(['user_id', 'recipe_id', 'date', 'rating', 'review'])
            for row in reader:
                if int(row[0]) in allowed_ids and int(row[1]) in allowed_ids:
                    writer.writerow(row)

    def simrank(self, decay_factor: float = 0.9, error_threshold: float = 1e-3, pplus: bool = False,
                file_path: str = 'data/simrank.json') -> None:
        """Calculates the Simrank similarity between every pair of vertices in the graph, using the given decay_factor.
        Stops iterating when the difference between the calculated similarity is less than the error_threshold. If
        pplus is True, isntead calculate the Simrank++ similarity between every pair of vertices. The file_path
        specifies the location to store the computed similarity scores to reduce future time complexity.

        Preconditions:
            - 0.0 <= decay_factor <= 1.0
            - error_threshold > 0.0
        """
        error = float('inf')
        score_type = 'simrank' + pplus * '_pplus'
        with tqdm(total=100) as pbar:
            while error > error_threshold:
                new_simrank = {}
                for u in self._vertices.values():
                    for v in self._vertices.values():
                        new_simrank[(u, v, False)] = u.calculate_simrank(v, decay_factor, pplus, sentimental=False)
                        new_simrank[(u, v, True)] = u.calculate_simrank(v, decay_factor, pplus, sentimental=True)
                error = 0
                for (u, v, sentimental), score in new_simrank.items():
                    error += abs(u.similarity_scores[sentimental][score_type].get(v, 0.0) - score)
                    u.similarity_scores[sentimental][score_type][v] = score
                    v.similarity_scores[sentimental][score_type][u] = score
                error = error / len(new_simrank)
                pbar.update(int(error_threshold / error * 100 - pbar.n))
            pbar.update(100 - pbar.n)
        simrank_data = {
            s: {x.id: {w.id: w_score for w, w_score in x.similarity_scores[s][score_type].items()}
                for x in self._vertices.values()} for s in {True, False}
        }
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(simrank_data, file)

    def pagerank(self, damping_factor: float = 0.85, error_threshold: float = 1e-6,
                 file_path: str = 'data/pagerank.json') -> None:
        """Calculates the Personalized Pagerank similarity between every pair of vertices in the graph, using the given
        damping_factor. The error_threshoold is passed into the random_walk function to determine the maximum
        acceptable difference between the calculated similarity value and the convergent value. The file_path specifies
        the location to store the computed similarity scores to reduce future time complexity.

        Preconditions:
            - 0.0 <= damping_factor <= 1.0
            - error_threshold > 0.0
        """
        score_type = 'pagerank'
        for v in tqdm(self._vertices.values()):
            v.similarity_scores[True][score_type] = v.random_walk(set(self._vertices.values()), damping_factor,
                                                                  error_threshold, True)
            v.similarity_scores[False][score_type] = v.random_walk(set(self._vertices.values()), damping_factor,
                                                                   error_threshold, False)
        pagerank_data = {
            sentimental: {x.id: {u.id: score for u, score in x.similarity_scores[sentimental][score_type].items()} for
                          x in self._vertices.values()} for sentimental in {True, False}
        }
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(pagerank_data, file)

    def load_precomputed(self, score_type: str = 'simrank', file_path: str = 'data/simrank.json') -> None:
        """Loads the precomputed similarity scores from the specified file_path, depending on the type of score that is
        given in score_type, reduce future computational complexity.

        Preconditions:
            - score_type in {'simrank', 'simrank_pplus', 'pagerank'}
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            precomputed_data = json.load(file)
        for sentimental in {True, False}:
            sentimental_str = 'true' if sentimental else 'false'
            for vertex_id, vertex_simrank in precomputed_data[sentimental_str].items():
                score_dict = {self._vertices[int(v)]: score for v, score in vertex_simrank.items()}
                self._vertices[int(vertex_id)].similarity_scores[sentimental][score_type] = score_dict

    def get_similarity_score(self, item1: int, item2: int, score_type: str = 'jaccard',
                             sentimental: bool = False) -> float:
        """Return the similarity score between the two given items in this graph depending on similarity score type
        specified by score_type.

        Preconditions:
            - score_type in {'jaccard', 'overlap', 'cosine', 'tanimoto', 'simrank', 'simrank_pplus', 'pagerank'}
        """
        if item1 not in self._vertices or item2 not in self._vertices:
            raise ValueError
        score_functions = {
            'jaccard': _WeightedVertex.jaccard_sim,
            'overlap': _WeightedVertex.overlap_sim,
            'cosine': _WeightedVertex.cosine_sim,
            'tanimoto': _WeightedVertex.tanimoto_sim,
            'simrank': _WeightedVertex.simrank_sim,
            'simrank_pplus': _WeightedVertex.simrank_pplus_sim,
            'pagerank': _WeightedVertex.pagerank_sim
        }
        return score_functions[score_type](self._vertices[item1], self._vertices[item2], sentimental=sentimental)

    def recommend_recipes(self, recipe_name: str, limit: int = 10, score_type: str = 'jaccard',
                          sentimental: bool = False) -> (
            list[tuple[dict[str, Union[int, str, list[str], list[int], date]], str]]):
        """Return a list of up to limit recommended recipes based on similarity to the given recipe_name, and the
        specified similarity score_type. The returned list is a tuple containing the recipe's metadata followed by the
        a score bin which is either "Highly Recommended", "Moderately Recommended", or "Not Recommended", for the
        first, second, and third thirds of the sorted list of vertices based on their similarity scores, from highest
        to lowest. Raises a KeyError if the specified recipe_name is not the name of a recipe in the graph.

        Preconditions:
            - limit <= len(self._vertices)
            - score_type in {'jaccard', 'overlap', 'cosine', 'tanimoto', 'simrank', 'simrank_pplus', 'pagerank'}
        """
        recipe_list = []
        recipe_id = None
        for vertex_id, v in self._vertices.items():
            if v.is_recipe and recipe_name == v.metadata['name']:
                recipe_id = vertex_id
                break
        if not recipe_id:
            raise KeyError

        v = self._vertices[recipe_id]
        for other in self._vertices.values():
            score = self.get_similarity_score(v.id, other.id, score_type, sentimental)
            if other.is_recipe and score > 0 and other != v:
                # recipe_list.append((other.metadata['name'] if other.metadata else other.id, score))
                recipe_list.append((other.metadata if other.metadata else other.id, score))
        recipe_list.sort(key=lambda tup: -tup[1])
        medium_score, low_score = recipe_list[int(len(recipe_list) / 3)][1], \
            recipe_list[int(len(recipe_list) / 3 * 2)][1]
        recommendations = []
        for recipe_metadata, score in recipe_list:
            recommendations.append((recipe_metadata, "Highly Recommended" if score > medium_score else (
                "Moderately Recommended" if medium_score >= score > low_score else "Not Recommended")))
        return recommendations[:limit]


def load_weighted_review_graph(file_paths: dict[str, str] = None,
                               load_precomputed: bool = True, load_sentiment: bool = True) -> WeightedGraph:
    """Return a recipe review weighted graph corresponding to the Food website dataset. By default, a reduced version
    of the recipes.csv and interactions.csv dataset is used. These files, along with the location of the precomputed
    files can be specified in the file_paths dictinoary. If laod_precomputed is True, the specified Simrank, Simrank++,
    and Personalized Pagerank similarity scores are loaded into the graph.

    Preconditions:
        - all(key in {'interactions', 'recipes', 'simrank', 'simrank_pplus', 'pagerank'} for key in file_paths)
    """
    default_file_paths = {'interactions': 'data/interactions_small.csv',
                          'recipes': 'data/recipes_small.csv',
                          'simrank': 'data/simrank.json',
                          'simrank_pplus': 'data/simrank_pplus.json',
                          'pagerank': 'data/pagerank.json'}
    if file_paths is None:
        file_paths = default_file_paths
    for file_path, path in default_file_paths.items():
        if file_path not in file_paths:
            file_paths[file_path] = path

    graph = WeightedGraph()

    with open(file_paths['recipes'], 'r', encoding='utf-8') as recipes_file:
        reader = csv.reader(recipes_file)
        next(reader, None)
        for row in reader:
            recipe_id = int(row[1])
            metadata = {
                'name': row[0],
                'minutes': int(row[2]),
                'submission_date': date(*[int(x) for x in row[4].split('-')]),
                'tags': literal_eval(row[5]),
                'nutrition': literal_eval(row[6]),
                'steps': literal_eval(row[8]),
                'description': row[9],
                'ingredients': literal_eval(row[10])
            }
            graph.add_vertex(recipe_id, True, metadata)

    with open(file_paths['interactions'], 'r', encoding='utf-8') as interactions_file:
        reader = csv.reader(interactions_file)
        file_length = sum(1 for _row in reader) - 1

    with open(file_paths['interactions'], 'r', encoding='utf-8') as interactions_file:
        reader = csv.reader(interactions_file)
        next(reader, None)
        for row in tqdm(reader, total=file_length):
            graph.add_vertex(int(row[0]), False)
            interaction = Interaction(int(row[3]), row[4], date(*[int(x) for x in row[2].split('-')]), load_sentiment)
            graph.add_edge(int(row[0]), int(row[1]), interaction)
    if load_precomputed:
        graph.load_precomputed('simrank', file_path=file_paths['simrank'])
        graph.load_precomputed('simrank_pplus', file_path=file_paths['simrank_pplus'])
        graph.load_precomputed('pagerank', file_path=file_paths['pagerank'])
    return graph


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120,
        'extra-imports': ['csv', 'networkx', 'metadata', 'datetime', 'ast', 'random', 'tqdm', 'json'],
        'allowed-io': ['load_weighted_review_graph', 'WeightedGraph.simrank', 'WeightedGraph.pagerank',
                       'WeightedGraph.load_precomputed', 'WeightedGraph.densest_subgraph',
                       'WeightedGraph.generate_subgraph_dataset'],
    })
