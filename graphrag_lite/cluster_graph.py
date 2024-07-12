# Code modified from Microsoft/GraphRAG project

"""A module containing cluster_graph, apply_clustering, and run_layout methods definition."""
import asyncio
import logging
from enum import Enum
from random import Random
from typing import Any, Optional, cast

import networkx as nx
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_iterable
from graphrag.index import run_pipeline
from graphrag.index.config import PipelineWorkflowReference
from graphrag.index.graph.utils import stable_largest_connected_component
from graphrag.index.utils import gen_uuid, load_graph
from graspologic.partition import hierarchical_leiden

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Basic logging configuration
Communities = list[tuple[int, str, list[str]]]


def run_leiden(
    graph: nx.Graph, args: dict[str, Any]
) -> dict[int, dict[str, list[str]]]:
    """Run method definition."""
    max_cluster_size = args.get("max_cluster_size", 10)
    use_lcc = args.get("use_lcc", True)
    if args.get("verbose", False):
        log.info(
            "Running leiden with max_cluster_size=%s, lcc=%s", max_cluster_size, use_lcc
        )

    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    levels = args.get("levels") or sorted(node_id_to_community_map.keys())

    results_by_level: dict[int, dict[str, list[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = str(raw_community_id)
            result.setdefault(community_id, []).append(node_id)

    return results_by_level


def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: Optional[int] = 0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """Return Leiden root communities."""
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        level = partition.level
        results.setdefault(level, {})[partition.node] = partition.cluster

    return results


def cluster_graph(
    input: VerbInput,
    callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    level_to: Optional[str] = None,
    **_kwargs,
) -> TableContainer:
    """
    Apply a hierarchical clustering algorithm to a graph. The graph is expected to be in graphml format. The verb outputs a new column containing the clustered graph, and a new column containing the level of the graph.
    """
    output_df = cast(pd.DataFrame, input.get_input())
    var = output_df[column].apply(lambda graph: run_layout(strategy, graph))

    level_to = level_to or f"{to}_level"
    output_df[[to, level_to]] = [None, None]
    graph_level_pairs_column = []

    for _, row in progress_iterable(
        output_df.iterrows(), callbacks.progress, len(output_df)
    ):
        levels = list({level for level, _, _ in row[var]})
        row[level_to] = levels
        graph_level_pairs = [
            (
                level,
                "\n".join(
                    nx.generate_graphml(
                        apply_clustering(
                            cast(str, row[column]), cast(Communities, row[var]), level
                        )
                    )
                ),
            )
            for level in levels
        ]
        graph_level_pairs_column.append(graph_level_pairs)

    output_df[to] = graph_level_pairs_column
    output_df = output_df.explode(to, ignore_index=True)
    output_df[[level_to, to]] = pd.DataFrame(
        output_df[to].tolist(), index=output_df.index
    )
    output_df.drop(columns=[var], inplace=True)

    return TableContainer(table=output_df)


def apply_clustering(
    graphml: str, communities: Communities, level=0, seed=0xF001
) -> nx.Graph:
    """Apply clustering to a graphml string."""
    random = Random(seed)
    graph = nx.parse_graphml(graphml)
    for community_level, community_id, nodes in communities:
        if level == community_level:
            for node in nodes:
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level

    for node in graph.nodes:
        graph.nodes[node]["degree"] = int(graph.degree(node))
        graph.nodes[node]["human_readable_id"] = node
        graph.nodes[node]["id"] = str(gen_uuid(random))

    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level

    return graph


class GraphCommunityStrategyType(str, Enum):
    """GraphCommunityStrategyType class definition."""

    leiden = "leiden"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


def run_layout(
    strategy: dict[str, Any], graphml_or_graph: str | nx.Graph
) -> Communities:
    """Run layout method definition."""
    graph = load_graph(graphml_or_graph)
    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes")
        return []

    clusters: dict[int, dict[str, list[str]]] = {}
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)

    if strategy_type == GraphCommunityStrategyType.leiden:
        clusters = run_leiden(graph, strategy)
    else:
        raise ValueError(f"Unknown clustering strategy {strategy_type}")

    results: Communities = []
    for level, cluster_dict in clusters.items():
        for cluster_id, nodes in cluster_dict.items():
            results.append((level, cluster_id, nodes))

    return results


dataset = pd.read_parquet("data/create_summarized_entities.parquet")


async def run_python():
    """Run a pipeline using the python API"""
    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            steps=[
                {
                    "verb": "cluster_graph",
                    "args": {
                        "strategy": {"type": "leiden"},
                        "column": "entity_graph",
                        "to": "clustered_graph",
                        "level_to": "level",
                    },
                },
                {
                    "verb": "select",
                    "args": {
                        "columns": (["level", "clustered_graph"]),
                    },
                },
            ]
        ),
    ]

    tables = []
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)
    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        print(pipeline_result.result)
        pipeline_result.result.to_parquet("data/create_base_entity_graph.parquet")
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(run_python())
