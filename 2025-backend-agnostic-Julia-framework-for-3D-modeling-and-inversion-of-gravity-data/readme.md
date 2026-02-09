# Backend-agnostic Julia framework for 3D modeling and inversion of gravity data
[📄 arXiv:2602.03857](https://arxiv.org/abs/2602.03857)

## Authors

**Nimatullah** ([@naimat04](https://github.com/naimat04))  
Department of Earth Sciences, Indian Institute of Technology Bombay, India  

**Pankaj K. Mishra**  
Geological Survey of Finland  

**Jochen Kamm**  
Geological Survey of Finland  

**Anand Singh**  
Department of Earth Sciences, Indian Institute of Technology Bombay, India  

---

## Abstract

This paper presents a high-performance framework for three-dimensional gravity modeling and inversion implemented in Julia, addressing key challenges in geophysical modeling such as computational complexity, ill-posedness, and the non-uniqueness inherent to gravity inversion. The framework adopts a data-space inversion formulation to reduce the dimensionality of the problem, leading to significantly lower memory requirements and improved computational efficiency while maintaining inversion accuracy. Forward modeling and inversion operators are implemented within a backend-agnostic kernel abstraction, enabling execution on both multicore CPUs and GPU accelerators from a single code base. Performance analyses conducted on NVIDIA CUDA GPUs demonstrate substantial reductions in runtime relative to CPU execution, particularly for large-scale datasets involving up to approximately 3.3 million rectangular prisms, highlighting the scalability of the proposed approach. The inversion incorporates implicit model constraints through the data-space formulation and depth-weighted sensitivity, which mitigate depth-related amplitude decay and yield geologically coherent, high-resolution subsurface density models. Validation using synthetic models confirms the ability of the framework to accurately reconstruct complex subsurface structures such as vertical and dipping dykes. Application to field gravity data further demonstrates the robustness and practical utility of the GPU-accelerated framework, with the recovered models showing strong consistency with independent geological constraints and prior interpretations. Overall, this work underscores the potential of GPU-enabled computing in Julia to transform large-scale gravity inversion workflows, providing an efficient, extensible, and accurate computational solution for high-resolution geophysical studies.

---

## How to cite

If you use this work, please cite it as:

```bibtex
@article{Nimatullah2026GravityJulia,
  title   = {Backend-agnostic Julia framework for 3D modeling and inversion of gravity data},
  author  = {Nimatullah and Mishra, Pankaj K. and Kamm, Jochen and Singh, Anand},
  journal = {arXiv preprint arXiv:2602.03857},
  year    = {2026},
  archivePrefix = {arXiv},
  eprint  = {2602.03857},
  primaryClass = {physics.geo-ph}
}
