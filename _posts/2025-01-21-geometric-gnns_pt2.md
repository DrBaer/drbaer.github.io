---
layout: distill
title: Equivariant Graph Neural Networks Pt1
description: SE(3) Equivariant Graph Neural Networks with Cartesian Vectors - Step by Step
draft: false
tags:
giscus_comments: false
date: 2025-01-19
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true
# thumbnail:
# og_image:
# og_image_width: 2126
# og_image_height: 1478
authors:
  - name: Lars Rosenbaum
    url: "https://lcrosenbaum.github.io"
output: distill::distill_article
bibliography: geometric-gnns_pt2.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Equivariant Geometric GNNs - Basics

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---
This blog post is part of a series on geometric graph neural networks (GNNs) and their applications:<br />
[Part 1]({% post_url 2025-01-17-geometric-gnns_pt1 %}): Invariant GNNs and Alphafold 3<br />
:arrow_right:[Part 2]({% post_url 2025-01-21-geometric-gnns_pt2 %}): Equivariant GNNs with Cartesian Vectors

In part 1 we took a look at SE(3) invariant GNNs, their limitations and a way to overcome the limitations by going towards a fully connected 3-body GNN or equivalently multiplicative triangular updates . In this post, we'll explore the second option: using more informative features, more specifically equivariant features via cartesian vectors. We use the same notation as in part 1, unless additional notation is needed. Many of the topics covered in this post can be found in surveys <d-cite key="duval_hitchhikers_2023,zhang2023artificial"/>, but I try to bring together some concepts that were unclear to me even after reading those. As an additional cherry, we'll try to go beyond the standard ways of using transformers or message passing by using equivariant random walks on graphs.

*Again, why do we need equivariance?* The first part of the answer, as already mentioned is more expressiveness. By doing message passing with vector properties, we can show that one can transport e.g. 3-body information over the graph. Furthermore, we can represent information with respect to an external (directional) force field or achieve equivariance with respect to reflections (e.g. for chirality), which was not possible with distance matrix based approaches. The second part of the answer is, that while we can learn all conservative vector fields using invariant networks, this is not possible for non-conservative vector fields, which means we can also learn more target functions. Disclaimer: Most vector fields we are interested in are actually conservative.

### Equivariant Geometric GNNs - Basics

<div class="geometric_graph_equivariant">
  {% include figure.liquid path="assets/img/geometric-gnns_pt2/geometric_graph_equivariant.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig1" class="caption" style="text-align: left;">Fig. 1 Geometric graph with invariant and equivariant features. Under rotation and translation the node features $\mathbf{s}_i$ are invariant, the node positions $\vec{x}_i$ are rotated and translated, and the new vector features $\vec{\mathbf{v}}_i$  are just rotated.</div>
</div>

Fig. 1 shows the same graph as in part 1, but now each node $i$ has multiple cartesion vectors $\vec{\mathbf{v}}_i \in \mathbb{R}^{b \times 3}$, similar to how we have multiple channels $c$ for the scalars $\mathbf{s}_i$. We end up with a graph $$\mathcal{G} = (\vec{\mathbf{V}}, \mathbf{S}, \mathbf{A}, \vec{\mathbf{x}})$$, where $\vec{\mathbf{V}} \in \mathbb{R}^{n \times b \times 3}$. Each list of vectors $\vec{\mathbf{v}}_i$ transforms according to the SO(3) transformation matrix $\mathbf{R}$ when the graph is rotated by the same matrix: $$\vec{\mathbf{v}}_i' = \vec{\mathbf{v}}_i \mathbf{R}$$. At this point, you might already ask yourself, why exactly I chose 3-dimensional vectors and wether we could choose different dimensionalities of vectors instead. The answer to this question is yes, we can use $(2l+1)$-dimensional vectors $l \in \mathbb{N}$, e.g. as done by several works <d-cite key="thomas_tensor_2018,wang_enhancing_2024,simeon_tensornet_2024"/>. However, the answer goes beyond the basics, so let's defer that to later and start with the basics and 3D vectors.

*What are the building blocks?* To construct message-passing neural network layers, we can use any layers and functions we already used in part 1 on scalar messages, scalar node features, distances, and angles. As those features are invariant, we can use any non-linear function to transform them. For the vector features, we need to be more careful. We want to construct functions $f$ that fulfill the following equivariance property:

$$\begin{equation}
f(\vec{\mathbf{v}}_i \mathbf{R}) = f(\vec{\mathbf{v}}_i) \mathbf{R}
\end{equation}$$

Note that right multiplication of the rotation matrix is performed, as we defined our vectors to be row vectors. The following operations are equivariant or invariant given vector features $\vec{\mathbf{v}}, \vec{\mathbf{w}}$:
* Linear combination of equivariant functions, which includes multiplication of vectors with scalars: 
$$\begin{equation}\begin{split}
&\,a \cdot f_a (\vec{\mathbf{v}} \mathbf{R}) + b \cdot f_b(\vec{\mathbf{w}} \mathbf{R})\\
=& \,a \cdot f_a (\vec{\mathbf{v}} ) \mathbf{R} + b \cdot f_b(\vec{\mathbf{w}} )\mathbf{R}\\
=&\, \left( a \cdot f_a (\vec{\mathbf{v}} ) + b \cdot f_b(\vec{\mathbf{w}} ) \right) \mathbf{R}
\end{split}\end{equation}$$

* Linear combination of vectors (dense layer), with $$\mathbf{W}\in \mathbb{R}^{ b' \times b}$$:
$$\begin{equation}
\mathbf{W} \left( \vec{\mathbf{v}} \mathbf{R} \right)_k = \left( \mathbf{W} \vec{\mathbf{v}} \right) \mathbf{R}  
\end{equation}$$

* Contraction (scalar product), where $\vec{v}_k$ is the k-th row of $\vec{\mathbf{v}}$: 
$$\begin{equation}
\langle \vec{v}_k \mathbf{R}, \vec{w}_k \mathbf{R}\rangle = \vec{v}_k \mathbf{R} \mathbf{R}^T \vec{w}_k^T = \vec{v}_k \vec{w}_k^T = \langle \vec{v}_k, \vec{w}_k \rangle
\end{equation}$$

* Cross-product, with $\theta$ being the angle between and $\vec{\mathbf{n}}$ the normal vector of the plane spanned by $\vec{v}_k, \vec{w}_k$:
$$\begin{equation}
(\vec{v}_k \mathbf{R}) \times (\vec{w}_k \mathbf{R}) = \left( \|\vec{v}_k \mathbf{R}\| \|\vec{w}_k \mathbf{R}\| sin(\theta) \vec{\mathbf{n}} \right)\mathbf{R} = \left( (\vec{v}_k ) \times (\vec{w}_k) \right) \mathbf{R}
\end{equation}$$

All equivariant GNNs use a subset of these operations. Note that multiplication with scalars include gating mechanism, where scalars can be generated in arbitrarily complex ways. For dense layers, one has to be careful, wether one is using row or column vectors. Some publications, e-g VisNet<d-cite key="wang_enhancing_2024"/> (Supplements Eq. 65), use column vectors and then proof equivariant via $\mathbf{R} \mathbf{W} \vec{\mathbf{v}}_i =  \mathbf{W} \mathbf{R} \vec{\mathbf{v}}_i$, which is of course incorrect. Instead one has to do a right multiplication of $\mathbf{W}$, see e.g. <d-cite key="le2022representation"/>. 



To see why equivariant GNNs are more expressive even without full connectivity take the following example, which is quite close 






So let's look at a subset influental works: EGNN<d-cite key="satorras_e_2021"/>, GVP<d-cite key="jing_learning_2020"/>, PaiNN<d-cite key="schutt_equivariant_2021"/>, VisNet<d-cite key="wang_enhancing_2024"/>, and EQGAT<d-cite key="le2022representation"/>.

### Going Further - Higher Dimensional Cartesian Vectors

Look at TensorNet<d-cite key="simeon_tensornet_2023"/>.

### Going even Further - From Cartesian to Spherical Tensors


Tensor field networks<d-cite key="thomas_tensor_2018"/>, one of the earliest equivariant neural networks for 3d point clouds (including molecules) uses a fully connected graph, real spherical harmonics and the tensor product for message passing. Spherical harmonics are an orthonormal basis for functions on the sphere similar to how fourier series are a basis for functions on the circle. A very good introduction to those topics and spherical harmonics can be found in a blog post by Sophia Tang<d-cite key="tang_spherical_equivariant"/>. However, these networks have a drawback of high computational load as the tensor product needs to be broken down to irreducible representations with Clebsch Gordan coefficients. This means the networks do not scale very well to larger graphs. Furthermore, while these networks do have more power theoretically<d-cite key="joshi_expressive_2023usually"/>, this additional power is not needed for the common predictions we want to perform. In fact, we discussed in the last post that most of them are even invariant. Results from recent publications underline this observation: networks using spherical harmonics are on par with the simpler equivariant networks using cartesian tensors<d-cite key="wang_enhancing_2024,gasteiger_gemnet_2021"/>. Therefore, in this post, we will focus on the equivariant networks using cartesian tensors.









