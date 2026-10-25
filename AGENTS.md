Read README.md to understand the project.

* Design code for performance and maintainability on 2, 4, or more nodes. The project MUST also work on a single node, but single-node support is intended primarily for local testing.
* Single-node performance is not a priority. Focus performance optimizations on multi-node inference.

## NN

This project uses the NN engine for the neural network inference (placed in `src/nn`). Some main assumptions here:

* The model graph MUST be built in this way that every segments we synchronize data between node. Also this design can be used to synchronize GPU memory with CPU memory at the end of the segment.
