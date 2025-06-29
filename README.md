# Differentiable Probabilistic Logic Circuits Compiler

[![Unit and Integration Tests + Lint](https://github.com/mathjug/dPASP-Circuit-To-Network/actions/workflows/python-app.yml/badge.svg)](https://github.com/mathjug/dPASP-Circuit-To-Network/actions/workflows/python-app.yml)

A framework for compiling logical-probabilistic circuits, specifically **_sd-DNNFs_**, into differentiable neural network models for accelerated neuro-symbolic computation in _PyTorch_.

## Overview
This project focuses on the development and evaluation of strategies for compiling logical-probabilistic circuits, particularly those in _sd-DNNF_ (Smooth Deterministic Decomposable Negation Normal Form), into differentiable models compatible with frameworks like _PyTorch_.

The primary goal is to accelerate neuro-symbolic systems by transforming these symbolic circuits into optimized neural architectures. This transformation allows the models to leverage key features of modern deep learning, such as GPU parallelism and automatic differentiation, bridging the gap between symbolic reasoning and neural computation.

By creating an efficient and expandable foundation for these hybrid systems, this work aims to provide methodological contributions and practical implementations for research and application in areas like intelligent systems, robotics, and symbolic learning.

- - - -

## Project Goals & Compilation Strategies
The main objective is to investigate and implement different strategies for transforming _sd-DNNF_ circuits into neural network-based models. Each strategy is evaluated on performance, differentiability support, clarity, and extensibility.

### Trivial Approach
This method performs a direct, one-to-one substitution of logical gates with arithmetic operations.
- AND gates are mapped to tensor multiplication nodes (`torch.prod`).
- OR gates are mapped to tensor summation nodes (`torch.sum`).

The `RecursiveNN` and `IterativeNN` models in this repository are implementations of this strategy.

### Layered Neural Architecture
A more advanced strategy involving the creation of a layered neural network with alternating, dedicated layers for sum and product operations. This approach aims to optimize execution flow and resource usage on parallel hardware.

### Matrix-based Approach
This involves converting the entire circuit into a series of optimized vector and tensor operations, abstracting away the node-by-node structure to maximize throughput using the _PyTorch_ API.

- - - -

## Key Features
- **sd-DNNF Compilation**: Parses circuits in `.sdd` format and compiles them into differentiable computational graphs.
- **Probabilistic Inference**: Provides a high-level API (`QueryExecutor`) for calculating conditional probabilities P(Query | Evidence).
- **Multiple Compilation Strategies**: Implements and evaluates different approaches for circuit-to-network conversion.
- **Automatic Differentiation**: The resulting models are fully differentiable, enabling gradient-based analysis and training of neuro-symbolic models.
- **Comprehensive Test Suite**: Includes unit and integration tests to validate correctness, performance, and differentiability.

- - - -

## Installation
1. **Clone the repository**:

```bash
git clone https://github.com/mathjug/dPASP-Circuit-To-Network.git
cd dPASP-Circuit-To-Network
```

2. **Install dependencies**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

- - - -

## Testing
The project includes a robust test suite using `pytest` to ensure the correctness and reliability of the implemented features.

- **Unit Tests**: Test individual components in isolation, mocking dependencies.
- **Integration Tests**: Test the entire pipeline from file parsing to final query result using real components.

To run all tests, navigate to the project's root directory and execute:

```bash
pytest
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
