---
layout: distill
title: Invariant Graph Neural Networks and Alfa Fold 3
description: A small journey from Invariant Graph Neural Networks to Alfa Fold 3 Pairformer and back.
draft: false
tags:
giscus_comments: false
date: 2025-01-17
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
bibliography: geometric-gnns_pt1.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Invariant Geometric GNNs
  - name: Fully Connected Graphs and Transformers
  - name: Alfafold 3 Pairformer
  - name: A different view on Pairformer
  - name: Some Performance Evaluations
  - name: Final Thoughts

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
Whether predicting molecular properties or classifying objects in a point cloud, many tasks in science and engineering involve data invariant to rotation and translation. This is where invariant graph neural networks (GNNs) excel. In this post, we will explore the limitations of different invariant GNNs and connect them to AlphaFold 3's Pairformer module. While my current work involves analyzing sensor data point clouds, the principles of invariance are broadly applicable, including in AlphaFold 3's prediction of molecular structures.

We will specifically examine the Pairformer's "triangle updates" and draw a connection between them and a specific type of invariant GNN. This offers a fresh perspective on the Pairformer's effectiveness, going beyond the original paper's explanation via the triangle inequality. A deep dive into AlphaFold 3 can be found in a blog post by Elana Simon's <d-cite key="simon_illustrated_alphafold" />, and for a geometric GNN survey, see Duval et al. <d-cite key="duval_hitchhikers_2023" />. I'll follow the notation of Duval et al. in this post, which aligns with PyTorch Geometric's `flow="target_to_source"` inr `propagate()`.

While the positions are equivariant, many crucial outputs in molecular modeling are invariant to 3D rotations and translations (SE(3) transformations). For instance energy associated with a particular conformation of a molecule remains the same regardless of its orientation in space. Similarly, quantum chemical properties like the HOMO-LUMO gap or internal energy are intrinsic to the molecule's structure and don't change with its pose. While forces used in molecular dynamics simulations are equivariant (they change predictably with rotation/translation), they can be derived from the invariant energy. Even in complex tasks like protein folding or molecular docking, the ultimate goal is to determine the relative positions of atoms, often represented by a distance matrix, which is itself invariant to SE(3) transformations. The same principles apply to point cloud analysis from sensors, where e.g. classifying objects is an invariant task. 

**So for the rest of this post we assume that our prediction target is invariant to roto-translations in SE(3)**.  We will also assume that our prediction targets are permutation invariant, meaning the order of points or atoms in the input does not change the prediction target.

### Invariant Geometric GNNs

Given that our target predictions are invariant to rotations and translations, it's natural to design graph neural networks that respect this fundamental symmetry. This leads us to invariant geometric graph neural networks (GNNs), a powerful class of models that includes architectures like SchNet<d-cite key="schutt_schnet_2017"/>, DimeNet++ <d-cite key="gasteiger2020directional,dimenetpp"/>, or GemNet<d-cite key="gasteiger2021gemnet"/>. In this section we'll explore this type of GNNs and also their limitations.

<div class="geometric_graph_invariant">
  {% include figure.liquid path="assets/img/geometric-gnns_pt1/geometric_graph_invariant.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig1" class="caption" style="text-align: left;">Fig. 1 Geometric graph with invariant features. Under rotation and translation the node features $\mathbf{s}_i$ are invariant, and the node positions $\vec{x}_i$ are rotated and translated.</div>
</div>

*Invariant graph neural networks* (GNNs) are described by a geometric graph $$\mathcal{G} = (\mathbf{S}, \mathbf{A}, \vec{\mathbf{x}})$$ with nodes $$\mathcal{V}_{\mathcal{G}} = \{1, \ldots, n\}$$, node scalar feature matrix $$\mathbf{S} \in \mathbb{R}^{n\mathrm{x}c}$$, adjacency matrix $$\mathbf{A} \in \mathbb{R}^{n\mathrm{x}n}$$, and atom positions $$\vec{\mathbf{x}} \in \mathbb{R}^{n\mathrm{x}3}$$. Each node $$i$$ has node features $$\mathbf{s}_{i} \in \mathbb{R}^{c}$$ and a position $$\vec{x}_{i} \in \mathbb{R}^{3}$$.  The features $$\mathbf{s}_{i}$$ are invariant under transformations in SE(3). The adjacency matrix $$\mathbf{A}$$ defines the neighborhood $$\mathcal{N}_i = \{ j \in \mathcal{V}_{\mathcal{G}}\setminus i  \vert a_{ij} \neq 0 \}$$.

<div class="gnn_overview">
  {% include figure.liquid path="assets/img/geometric-gnns_pt1/gnn_overview.svg" class="img-fluid rounded z-depth-1" %}
  <div id="fig2" class="caption" style="text-align: left;">Fig. 2 General network architecture of geometric GNNs.</div>
</div>

*Message passing with 2-body messages*. The core of the GNN architecture is the interaction block, which updates messages $$\mathbf{m}_{ij}$$ and then uses the updated messages from neighbors $$j \in \mathcal{N}_i$$ to update node $$i$$. The messages are updated based on invariant information like node scalars $$\mathbf{s}_i, \mathbf{s}_j$$ and invariants calculated from the node positions $$\vec{x}_i, \vec{x}_j$$. E.g. one of the early architectures (SchNet <d-cite key="schutt_schnet_2017"/>) used radial basis function embeddings based on the distance between nodes.  $$d_{ij}=\|\vec{x}_{ij}\|=\|\vec{x}_i - \vec{x}_j\|$$. The interaction block performs the following calculations:

$$\begin{align}
\mathbf{m}^{t+1}_{ij} &= msg\_upd(\mathbf{s}^t_i, \mathbf{s}^t_j, d_{ij}, \mathbf{m}_{ij}^t) \label{eq:2bodymes}\\
\mathbf{s}^{t+1}_i &= node\_upd(\mathbf{s}_{i}^{t}, \bigoplus_{j \in \mathcal{N}_i} \mathbf{m}^{t+1}_{ij}) \label{eq:2bodyupd}\,
\end{align}$$

where $$msg\_upd, node\_upd$$ are non-linear functions and $$\bigoplus$$ is a permutation invariant aggregation, because the update should not depend on the order in which we process the neighbors. The choice of $$\bigoplus$$ is important as shown by Corso et al.<d-cite key="corso2020principal"/>, but for simplicity we will use $$\sum$$ in the remainder. But keep in mind that depending on the aggregation function the expressivity might be limited. The initial node features $$\mathbf{s}_i^0$$ are based on a learned embedding of input atom features $$\mathbf{z}_i$$. The learnt initial node features $$\mathbf{s}_i^0, \mathbf{s}_j^0$$, are then combined to initial messages $$\mathbf{m}_{ij}^0$$ together with the distance embeddings.

One can quickly find simple counter examples <d-cite key="joshi_expressive_2023,pozdnyakov2022incompleteness"/>, which show that two graphs with different invariant properties (e.g. area of enclosing bounding box) are not distinguishable by the simple messaging scheme of SchNet. The messages $$\mathbf{m}_{ij}$$ in Eq. \ref{eq:2bodymes} are called 2-body or 1-hop messages, as these include information (distances) of two nodes or "bodies", which is a hop over 1 edge. One can implement more powerful networks using the distances and angles between k-bodies, e.g. for 3-bodies (2-hop neighbors) using the angles $$\angle ijk = \angle (\vec{x}_{ij}, \vec{x}_{ik}) = \langle \frac{\vec{x}_{ij}}{d_{ij}} , \frac{\vec{x}_{ik}}{d_{ik}} \rangle$$. This can be further extended as shown in Fig. 3. Usually, this is not extended beyond 4-bodies, due to the increasing computational costs.

<div class="k-body">
  {% include figure.liquid loading="eager" path="assets/img/geometric-gnns_pt1/multi_body_gnn.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig3" class="caption" style="text-align: left;">Fig. 3 Variants of multi-body GNNs, where each variant also uses the information from the previous one: a) 2-body using only distances <d-cite key="schutt_schnet_2017"/>; b) 3-body using distances and angles between 3 bodies <d-cite key="gasteiger2020directional,gasteiger2021gemnet"/>; c) 4-body using distances, angles between 3 bodies and dihedral angles <d-cite key="gasteiger2021gemnet"/>. The uppper row visualizes visualizes the information used to calculate $m_{ij}^{t+1}$. The second and third row shows two graphs $\mathcal{G}_1, \mathcal{G}_2$, which cannot be distinguisghed with the message information used in this column (examples from Joshi et al.<d-cite key="joshi_expressive_2023"/>).</div>
</div>

*Going beyond 2-body messages*. In the following equations, I slightly diverge from the presentation in Duval et al.<d-cite key="duval_hitchhikers_2023" />, because the original publications by Gasteiger et al.<d-cite key="gasteiger2020directional,gasteiger2021gemnet"/> also use the previous layer messages $$\mathbf{m}_{ij}^{t}$$. Actually the previous messages are passed through an MLP, but I leave that out for simplicity. When comparing the equations with the original publications, please note that indexing differs. In the paper the `flow="source_to_target"` notation is used: $$\mathbf{m}_{ji}$$ for messages towards node $$i$$. All invariant GNNs presented in Fig. 3 use same node and edge updates and only differ in calculation of $$\hat{\mathbf{m}}_{ij}$$:
$$\begin{align}
\tilde{\mathbf{m}}_{ij} &= \mathbf{m}_{ij}^t + \hat{\mathbf{m}}_{ij} + \hat{\mathbf{m}}_{ji} \label{eq:edge_symmetry}  \\
\mathbf{s}^{t+1}_i    &= \mathbf{s}_{i}^{t} +  \sum_{j \in \mathcal{N}_i} f_{2-body}(d_{ij}) \odot \tilde{\mathbf{m}}_{ij} \label{eq:node_upd} \\
\mathbf{m}^{t+1}_{ij} &= \tilde{\mathbf{m}}_{ij} + g(\tilde{\mathbf{m}}_{ij},\mathbf{s}_{i}^{t+1}, \mathbf{s}_{j}^{t+1} ) \label{eq:edge_upd}
\end{align}$$

* a) SchNet <d-cite key="schutt_schnet_2017"/> like:

$$\begin{equation}
\tilde{\mathbf{m}}_{ij} = \mathbf{m}^{t}_{ij} \\
\end{equation}$$

* b) GemNet-T <d-cite key="gasteiger2021gemnet"/>, similar to DimeNet++<d-cite key="dimenetpp"/>:

$$\begin{equation}
\hat{\mathbf{m}}_{ij}^{triplet} = \sum_{k \in \mathcal{N}_i \setminus j} f_{3-body}(d_{ik}, \angle ijk )  \odot a(\mathbf{m}^{t}_{ik}) \label{eq:msg_gemnet_t}
\end{equation}$$

* c) GemNet-Q <d-cite key="gasteiger2021gemnet"/>

$$\begin{equation}\label{eq:msg_gemnet_q} 
\begin{split}
\hat{\mathbf{m}}_{ij}^{dihedral} = & \hat{\mathbf{m}}_{ij}^{triplet} + \\
 \sum_{\substack{m \in \mathcal{N}_i \setminus j\\n \in \mathcal{N}_m \setminus i,j\\ }} [ & f_{4-body}(d_{ij},\angle ijm , \angle ijmn)^T \mathbf{W} f_{3-body}(d_{im}, \angle imn)  \\
  & \odot f_{2-body}(d_{mn}) \\
  & \odot \mathbf{m}^t_{mn} ]
\end{split}
\end{equation}$$

The computational cost of GemNet-Q's message interaction, represented by $$\hat{\mathbf{m}}_{ij}^{dihedral}$$ for GemNet-Q in Eq. \ref{eq:msg_gemnet_q}, is clearly quadratic with respect to the average number of neighbors. This quadratic scaling makes the approach of relying solely on invariant information quickly computationally infeasible for larger systems. While the limitations in distinguishing graphs $\mathcal{G}_1$ and $\mathcal{G}_2$ (as illustrated in Fig. 3) might be less significant for small molecules, they could pose a substantial challenge when dealing with longer, more complex protein chains.  To address this, two potential strategies emerge, and in this post, we will focus on the first:

1. Use a fully connected graph and ...?
2. Use more informative features, e.g. equivariant cartesian tensors as node/edge information, resulting in equivariant GNN <d-cite key="schutt_equivariant_2021,wang_enhancing_2024" />

### Fully Connected Graphs and Transformers 

Let's explore the first idea, employing fully connected graphs. These graphs are often used in universal approximation theorems<d-cite key="duval_hitchhikers_2023,dym2020universality"/> for GNNs, where aggregation occurs over all nodes. While incorporating spherical harmonics is crucial for representing functions on the sphere, we'll focus on the implications of full connectivity for now.

<div class="k-body">
{% include figure.liquid loading="eager" path="assets/img/geometric-gnns_pt1/Pozdnyakov_2022_fig4_distance_matrix.png" class="img-fluid rounded z-depth-1" zoomable=true %} 
<div id="fig4" class="caption" style="text-align: left;">Fig 4. Top: Distance matrices of counter-example by Pozdnyakov et al.<d-cite key="pozdnyakov2022incompleteness"/>. Differences are highlighted by red box. Bottom: Visualization why non-distinguishable structures $A^+,A^-$ can pose a problem in ML.</div></div>

*Is a fully connected graph enough?* Fig. 4(top) shows the distance matrices of a counter-example by Pozdnyakov et al.<d-cite key="pozdnyakov2022incompleteness"/>. We can permute rows and columns of these matrices as the distances are independent of the order in which we list the points (recall that our prediction target is permutation invariant). However, we cannot transform the matrix of $A^-$ to $A^+$ by row/column permutations alone, due to the area marked by the red box. Thus the matrices are *distinct*. A 2-body GNN (e.g. SchNet) on a fully connected graph operates solely on the distances within a row of the distance matrix. If we strip away all the learnt features, we are left with an output that is dependent on the specific distance information used by the 2-body GNN:

$$\begin{equation}\label{eq:2body_power} 
 f_{2-body} = \bigoplus_{i} \bigoplus_{j \in \mathcal{N}_i} f( d_{ij}) = f( \{ \{d_{ij}\}_{j, j \neq i}  \, \}_i) 
\end{equation}$$

where $$\bigoplus_{j \in \mathcal{N}_i}$$ is due to the aggregation operation in Eq. \ref{eq:node_upd} and $$\bigoplus_i$$ is due to an assumption that we aggregate the information of all nodes for the final output. As long as the sets $$\{d_{ij}\}_{j, j \neq i}$$ contain the same values for all $i$, the 2-body GNN cannot distinguish the distance matrices. Consequently, it cannot distinguish $A^+$ and $A^-$. This limitation can significantly hinder learning, not only for this specific counter-example but also for nearby structures due to the smoothness of predictions. Fig 4. (bottom) illustrates this issue. Non-distiniguishable structures $A^+$ and $A^-$ are forced to have the same predicted value $\tilde{y}_{ML}$. Given that ML predictors are typically smooth to some extend, the neighboring structures on the data manifold are also affected. 

*What is a good criterion?* The distance matrix uniquely identifies a point cloud up to roto-translations and reflections<d-cite key="dokmanic2015euclidean"/>. This property stems from the fact that the Gram matrix $\mathbf{G}$, where $$G_{ij} = \langle \vec{x}_i, \vec{x}_j \rangle$$, can be calculated from the following relation $$ G_{ij} = \frac{1}{2} ( d_{ik}^2 + d_{jk}^2 - d_{ij}^2  )$$, where $k$ is an arbitrarily chosen reference point. We need a reference point because the distance matrix is translation invariant, while the Gram matrix is not. Singular value decomposition allows us to uniquely reconstruct the point positions $\vec{\mathbf{x}}$  (up to rotation and reflection) from the Gram matrix. Thus, if we can disregard enantiomers and assume our target function's atom properties are invariant under SO(3), the full distance matrix appears to be a suitable starting point. If our model can distinguish differences in distance matrices (and consequently, Gram matrices), we should be in good shape.

*How can we construct such models?* Lemma 1 of Villar et al. <d-cite key="villar2021scalars"/> demonstrates that a function based on the gram matrix and node features, $$f(\mathbf{G}, \mathbf{S})$$, is sufficient. While they provide an example using an MLP, this isn't ideal for our purpose because we need a model that a) generalizes across different input sizes and b) is permutation invariant. The counter-example in Fig. 4 is distinguishable by the 3-body networks described in Eq. \ref{eq:msg_gemnet_t}. To see this, let's perform a similar analysis as in Eq. \ref{eq:2body_power}, using the law of cosines to replace angles with distances.

$$\begin{equation}\label{eq:3body_power}
\begin{split}
 f_{3-body} &= \bigoplus_{i} \bigoplus_{j \in \mathcal{N}_i} \bigoplus_{k \in \mathcal{N}_i \setminus j} f( (d_{ij}, d_{ik}, \angle ijk) ) \\
  &= \bigoplus_{i} \bigoplus_{j \in \mathcal{N}_i} \bigoplus_{k \in \mathcal{N}_i \setminus j} f( (d_{ij}, d_{ik}, d_{jk}) )\qquad \text{#Law of cosines}\\
   &= f( \{ \{ \{(d_{ij},d_{ik},d_{jk}) \}_{k,k\neq{j},i} \}_{j,j\neq i} \}_i ) 
\end{split}\end{equation}$$

<div class="k-body">
{% include figure.liquid loading="eager" path="assets/img/geometric-gnns_pt1/3body_power.png" class="img-fluid rounded z-depth-1" %} 
<div id="fig5" class="caption" style="text-align: left;">Fig 5. Illustration of different triplets in distance matrices of the counter-example shown in Fig. 4. The shown triplet of $A^+$ cannot be matched to a 3-body message triplet of $A^-.</div></div>

Assuming a sufficiently powerful set function, we can distinguish distance matrices $A^+$ and $A^-$ if a set of non-matching triplets exists. An example triplet is illustrated in Fig 5. With a fully connected graph, these networks can also distinguish the graphs $\mathcal{G}_1, \mathcal{G}_2$ of Fig. 3. The 3-body invariant GNNs on fully connected graphs with multiple rounds of message passing are powerful enough to handle the counter-examples in Figures 1 and 4 of Pozdnyakov et al.<d-cite key="pozdnyakov2022incompleteness"/>. Whether this allows us to distinguish *all* distinct distance matrices or if counter-examples still exist is an interesting open question. This approach implies using Eq. \ref{eq:node_upd}. \ref{eq:edge_upd}, and \ref{eq:msg_gemnet_t}, but with the neighborhood $\mathcal{N}_i$ encompassing all other nodes. Of course, we can improve upon simple summation in Eq. \ref{eq:node_upd} by employing more sophisticated aggregation mechanisms, such as those found in transformers or the those proposed in Corso et al.<d-cite key="corso2020principal"/>. This is precisely what AlphaFold's Pairformer module does.

### Alfafold 3 Pairformer

<div class="l-body">
  {% include figure.liquid path="assets/img/geometric-gnns_pt1/alphafold3_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig6" class="caption" style="text-align: left;">Fig. 6 Structural overview from AlphaFold 3<d-cite key="abramson_accurate_2024" />.</div>
</div>

The Pairformer is the main representation learning module of AlfaFold 3. In the following, I will differ in notation of AlfaFold 3 and leave out some details like layer norm. The input to the Pairformer are the representations for pair $$\mathbf{m}_{ij} \in \mathbb{R}^{c_{pair}}$$ and single $$\mathbf{s}_{i} \in \mathbb{R}^{c_{single}}$$ with $$i,j \in \{1, ..., N_{Tokens}\}$$. Here I already explicitly used the symbols $$\mathbf{m}_{ij}$$ and $$\mathbf{s}_{i}$$ to stay in the GNN jargon for messages and node features, respectively. Each token can represent different types of entities (whole amino acid, whole nucleotide, single atom) but for our discussion this is mainly irrelevant. We could think of each token representing a single atom in a molecule. The pair represenation also contains informations from templates and an mutliple sequence alignment (MSA), but AlphaFold3 can be run without, so let's assume we do that.

The inputs of the Pairformer $$\mathbf{m}_{ij}, \mathbf{s}_{i}$$ are created from the input embedding $$\mathbf{s_i^{input}}$$, which are calculated in the "Input imbedder" block based on network inputs. The input embedding contains SO(3) invariant features that contain information on atom features and on pairwise distances between atoms within one amino acid, ligand, or an nucleic acid. The inputs to pairformer are created as 

$$
\begin{aligned}
\mathbf{m}_{ij} &= f_m(\mathbf{s_i^{input}}) + f_m(\mathbf{s_j^{input}}) + rpe(i,j) \\
\mathbf{s}_{i}  &= f_s(\mathbf{s_i^{input}})
\end{aligned}
$$ 

where $$rpe(i,j)$$ is the relative positional encoding. *Interestingly, the $rpe$ encodes (beside other parts) the token index distance (L. 6 Algorithm 3 Supplement <d-cite key="abramson_accurate_2024" />), which makes sense for protein sequences, but not for the atoms of the ligand, as it breaks the permutation invariance*. The distance of tokens now depends how we read in the atoms of the ligand at the network input. If we want to make a connection towards graph neural networks, we have to assume, that $rpe$ is actually an embedding of atom distances like the radial basis functions used in SchNet or GemNet.

<div class="l-body">
  {% include figure.liquid path="assets/img/geometric-gnns_pt1/alphafold3_pairformer_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig7" class="caption" style="text-align: left;">Fig. 7 Pairformer module overview from AlphaFold 3<d-cite key="abramson_accurate_2024" />.</div>
</div>

The main action on the pair representation within Pairformer are the triangle updates and triangle self-attention. A lot is already explained on the motivation of these updates in Elana's post<d-cite key="simon_illustrated_alphafold" />, so I just want to emphasize here that it does not enforce the triangle inequality but it is only an inductive bias. The modules perform the following updates, where functions $a, b, g, q, k \ldots$ can be different for each module:

* Triangular update "outgoing" edges: 
$$\begin{equation}\hat{\mathbf{m}}_{ij}^{outgoing} = g(\mathbf{m}_{ij}^t) \odot \sum_k a(\mathbf{m}_{ik}^t) \odot b(\mathbf{m}_{jk}^t)\end{equation}$$

* Triangular update "incoming" edges: 
$$\begin{equation}\hat{\mathbf{m}}_{ij}^{incoming} = g(\mathbf{m}_{ij}^t) \odot \sum_k a(\mathbf{m}_{ki}^t) \odot b(\mathbf{m}_{kj}^t)\end{equation}$$

* Triangular attention starting node: 
$$\begin{equation}
\begin{split}
\alpha_{ijk} &= softmax_k(\frac{1}{\sqrt{c}}q(\mathbf{m}_{ij}^t)^T k(\mathbf{m}_{ik}^t) + b(\mathbf{m}_{jk}^t)) \\
\hat{\mathbf{m}}_{ij}^{start} &= g(\mathbf{m}_{ij}^t) \odot \sum_k a_{ijk} \, v(\mathbf{m}_{ik}^t) 
\end{split}
\end{equation}$$

* Triangular attention ending node:
$$\begin{equation}
\begin{split}
\alpha_{ijk} &= softmax_k(\frac{1}{\sqrt{c}}q(\mathbf{m}_{ij}^t)^T k(\mathbf{m}_{kj}^t) + b(\mathbf{m}_{ki}^t)) \\
\hat{\mathbf{m}}_{ij}^{end} &= g(\mathbf{m}_{ij}^t) \odot \sum_k a_{ijk} \, v(\mathbf{m}_{kj}^t) 
\end{split}
\end{equation}$$

The above attention updates ignore the fact that we have multiple attention heads, but this only affects the channel dimension. The main action on the single representation is the single attention with pair bias:

$$\begin{align}
\alpha_{ij} &= softmax_j(\frac{1}{\sqrt{c}}q(\mathbf{s}_i^t)^T k(\mathbf{s}_j^t)+b(\mathbf{m}_{ij}^{t+1}))  \\
\mathbf{s}_i^{t+1} &= \mathbf{s}_i^t + g(\mathbf{s}_i^t) \odot \sum_j \alpha_{ij} \, v(\mathbf{s}_j^t)
\end{align}$$

where the functions $q, k, v$ are the query, key, and value vectors of transformers, $b$ is the bias function based on the pairs, and $g$ is a gating function as before for the pairs. Similar node updats are also used in more recent GNN architectures<d-cite key="wang_enhancing_2024,luo_one_2022" />

###  A different view on Pairformer

Let's rewrite the 3-body variant GemNet-T (Eq. \ref{eq:msg_gemnet_t}) for a fully connected graph and switch to using only distances due to the law of cosines, just as we did before in Eq. \ref{eq:3body_power}. Furthermore, instead of multiplying the distance and angle embeddings in every message update, we could also simply use the respective messages, as we already encoded the distances in the embedding of the messages:

$$\begin{equation}\begin{split}
\hat{\mathbf{m}}_{ij}^{triplet} &= \sum_{k} f_{2-body}(d_{ij}) \odot f_{3-body}(d_{ik}, \angle ijk ) \odot a(\mathbf{m}^{t}_{ik}) \\
&= \sum_{k} f_{3-body}(d_{ij}, d_{ik}, d_{jk}) \odot a(\mathbf{m}^{t}_{ik}) \qquad \text{#Law of cosines}\\
&= \sum_{k} a(\mathbf{m}^{t}_{ik}) \odot g(\mathbf{m}^{t}_{ij}) \odot b(\mathbf{m}^{t}_{jk}) \qquad \text{#Distances embedded in messages}\\ 
& = g(\mathbf{m}^{t}_{ij}) \odot \sum_{k} a(\mathbf{m}^{t}_{ik}) \odot b(\mathbf{m}^{t}_{jk}) \\
& = \hat{\mathbf{m}}_{ij}^{outgoing} \label{eq:msg_gemnet-t-distances}
\end{split}\end{equation}$$ 

This means that the triangle update of Pairformer can be interpreted as an 3-body message in a fully connected invariant GNN. Due to the symmetry operation in Eq. \ref{eq:edge_symmetry} there is also a similarity to $$\hat{\mathbf{m}}_{ij}^{incoming}$$. Jumper et al.<d-cite key="jumper_highly_2021" /> found that either the triangle multiplicative update or the triangle attention was enough to achieve good performance, though both resulted in the best performance. As we now know that looking at triangles is actually equal to 3-body messages, and 3-body messages are more powerful discriminators as 2-body messages, this might be an explanation, why at least one was necessary to achieve good performance.

###  Some Performance Evaluations

Let's explore the costs associated with the 3-body GNN and the triangle updates of Pairformer. The Jupyter notebook for the experiments can be found in the following [Gist](https://gist.github.com/lcrosenbaum/9e6bc573cc219c6eca6478346c822339). In all the experiments I use `batch_size=4`, `channel_dim=128` and 50 function calls to estimate runtimes on an A100. The experiments focus on the central operations of the 3-body GNN and the multiplicative triangle update shown in the following snippet:

```python
def triangle_mult(m: Float[Tensor, 'b nnodes nnodes c']):
    return einsum(m, m, '... i k c, ... j k c -> ... i j c')

def graph_3body(m: Float[Tensor, 'batched_nedges c'],
                idx_ij: Int[Tensor, 'batched_ntriplets'], 
                idx_ik: Int[Tensor, 'batched_ntriplets']):
    m_ik = m[idx_ik]
    return scatter(m_ik, idx_ij, dim=0, dim_size=m.size(0), reduce='sum')
```
<div class="k-body">
<iframe src="/assets/plotly/geometric_gnns_pt1/runtime_vs_numnodes.html" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
<div id="fig8" class="caption" style="text-align: left;">Fig. 8: Runtime of both models with different node sizes using float32 and float16. The GNN uses an average node degree of 32.</div>
</div>

The triangle update is memory heavy due to the complete pair matrix and uses `einsum` to perform the summing operation. The 3-body GNN gets the sparse edges features as input and the indices of triplets. It has to perform a `gather`and `scatter` operation. The runtime performance of both model types are shown in Fig. 8. The quadratic runtime complexity of the triangle multiplicative update is clearly visible. Switching to float16 more than halves the runtime for the triangle update, whereas the 3-body GNN get's slower, as the scatter operations cannot profit from the float16 precision. Up to around 220 nodes the triangle update is actually faster than the quite sparse GNN. This a bit less than the median length of typical protein chains<d-cite key="nevers2023protein" />. For small molecules the triangle update is highly competitive. For large chains one could use block-wise message matrixes, which is exactly what Alfafold 3 does in its input embedding block.

<div class="k-body">
<iframe src="/assets/plotly/geometric_gnns_pt1/runtime_vs_avgnodedegree.html" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
<div id="fig9" class="caption" style="text-align: left;">Fig. 9: Runtime of 3-body GNN with different average node degrees. The graph underlying graph has 512 nodes. The blue dashed line shows the runtime of the multplicative triangle update for 512 nodes.</div>
</div>

In another experiment I estimated the runtime of the 3-body GNN on a graph with 512 nodes for different average node degrees. The results in Fig. 9 indicate that the runtime scales quadratically with the average node degree, so we have to limit expressivity if we want competitive runtimes.

###  Final Thoughts

Invariant GNNs have limitations, but they can be really effective and are relatively lightweight in compute. The success of Alfafold 3 supports this, as it has actually the same power as 3-body GNNs. The question if fully connected 3-body GNNs can distinguish all distance matrices is an interesting one. A big drawback of the triangle update is the size of the pair matrix, which severely limits the scalability. 

As an alternative, one might perform local computations with 3-body GNNS and interleave with fully connected computations like triangle updates. This is exactly what we usually do in radar point cloud processing, as the data is quite sparse. We perform local computation within clusters, e.g. with sparse convolutions and exchange information globally via transformers.

So the post got long enough, but I hope you enjoyed it. In my next blog post, I want to explore the second possibility for getting more expressivity of the networks: using equivariant features<d-cite key="wang_enhancing_2024" />.