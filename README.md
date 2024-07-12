# GraphRAG Lite

A lightweight version of the Microsoft [GraphRAG](https://github.com/microsoft/graphrag) project.

> Note: The development is currently in progress.

## Getting Started

```shell
pip install -r requirements.txt
```

### 1. Graph Clustering

```python
python graphrag_lite/cluster_graph.py
```

This will create a `create_base_entity_graph.parquet` file in the `data` directory.

### 2. Global Search

Play with `graphrag_lite/global_search.ipynb` notebook.

- Configure the OpenAI API key and Model first
- This demo will cost approximately 20K tokens, around $0.1
