#### Example 1:

User Intent:
"I need a high-performance network slice with 3 main nodes connected in a star topology. The central node should handle most of the computation, while the links require moderate bandwidth."

Output:

```python
[
(
    "DeploymentIntent",
    "deploy_slice",
    {
        gml:"graph [
              description "high-performance star network"
              num_nodes 3
              node [
                id 0
                label "0"
                cpu 45
              ]
              node [
                id 1
                label "1"
                cpu 25
              ]
              node [
                id 2
                label "2"
                cpu 25
              ]
              edge [
                source 0
                target 1
                bw 10
              ]
              edge [
                source 0
                target 2
                bw 10
              ]
            ]",
    },
)
]
```
#### Explanation for Example 1:

- The user wants to create such a network slice, so it is DeploymentIntent and we should do the "deploy_slice" to realize his intent, for this action, we should write a gml file.
- The user requested a star topology with a high-performance central node
- CPU resources were allocated to the central node (id=0, cpu=45 units) and evenly distributed among the leaf nodes (25 units each), aligning with the user's desire of "high-performance."
- Links were assigned moderate bandwidth (10 units each), aligning with the user’s description of "moderate bandwidth."



#### Example 2:

User Intent:
"delete this slice: qa21ew11"

Output:

```python
[
(
    "ModificationIntent",
    "remove_slice",
    {
        slice_id:"qa21ew11",
    },
)
]
```

#### Explanation for Example 2:

- The user wants to delete the slice qa21ew11, so it is DeploymentIntent and we should do the "remove_slice" to realize his intent, for this action, we need the slice's id.
- User did give us the slice' id, it is qa21ew11.