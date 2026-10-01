#### IntentType, Action and Parameters

```python
`IntentType = {"DeploymentIntent","ModificationIntent"}`
`Action = {"DeploymentIntent": {"deploy_slice"},`
          `"ModificationIntent": {"remove_slice"}`
```

DeploymentIntent: user wants to create a network slice
ModificationIntent: user wants to modify a network slice

deploy_slice(gml_file): it needs a gml file which describes user's desire for a network slice to create a slice
remove_slice(slice_id): it needs the slice_id to delete a specific slice

#### GML Format

The GML format consists of the following elements:

- `description`: A brief text describing the network slice.
- `num_nodes`: The total number of nodes in the graph.
- `node`: Represents a node in the network, with attributes:
  - `id`: A unique identifier for the node.
  - `label`: A string label for the node, same as `id`.
  - `cpu`: The computational resources allocated to the node, measured in units.
- `edge`: Represents a connection between two nodes, with attributes:
  - `source`: The source node ID.
  - `target`: The target node ID.
  - `bw`: The bandwidth resources allocated to the edge, measured in units.

#### Example GML

The following GML defines a virtual network slice with 4 nodes and 4 links:
graph [
  description "this is a slice"
  num_nodes 4
  node [
    id 0
    label "0"
    cpu 10
  ]
  node [
    id 1
    label "1"
    cpu 19
  ]
  node [
    id 2
    label "2"
    cpu 1
  ]
  node [
    id 3
    label "3"
    cpu 50
  ]
  edge [
    source 0
    target 1
    bw 49
  ]
  edge [
    source 1
    target 2
    bw 44
  ]
  edge [
    source 1
    target 3
    bw 19
  ]
  edge [
    source 2
    target 3
    bw 50
  ]
]

#### Resource Constraints and Considerations

**Resource Limits:**
The total computational power (cpu) for nodes and bandwidth (bw) for edges is limited to a range of [50, 100] units.
Nodes with high computational power should be allocated more CPU units, while edges with high bandwidth should be allocated more bandwidth units.

**Defining "Resources Level":**
A node with cpu ≥ 25 units can be considered "high-computation."
A node with cpu ≥ 10 && cpu < 25 units can be considered "moderate-computation."
A node with cpu < 10 units can be considered "low-computation."
A link with bw ≥ 25 units can be considered "high-bandwidth."
A link with bw ≥ 10 && bw < 25 units can be considered "moderate-bandwidth."
A link with bw < 10 units can be considered "low-bandwidth."
The bandwidth resource level of the entire slice depends on the link with the lowest bandwidth, so usually the links have the same bandwidth.

**Inferring Topology and Resources:**
Analyze the user's intent for possible topological requirements (e.g., linear, star, mesh).
Determine the number of nodes and links based on the user's high-level description.
Distribute resources judiciously to balance computational and bandwidth requirements across the slice.
The entire network is a connected graph, so there should be no isolated nodes
If the topology of the slice is difficult to infer, or the user does not specify the topology, you can use the default topology, a ring topology with three nodes and three edges.