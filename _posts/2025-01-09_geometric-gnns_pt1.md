---
layout: distill
title: Invariant Graph Neural Networks and Alfa Fold 3
description: A small journey from Pairformer (AlphaFold), over geometric vector percetrons (GVPs), to ViSNet for molecular graphs.
draft: true
tags:
giscus_comments: false
date: 2025-01-08
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
    affiliations:
      name: Robert Bosch GmbH
    url: "https://drbaer.github.io"
output: distill::distill_article
bibliography: 2025-01-09_geometric-gnns_pt1.bib

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
  - name: Connections Between Models
  - name: Conclusion

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
*In this post I am exploring the representation learning module (Pairformer) of AlphaFold 3<d-cite key="abramson_accurate_2024" /> and it's connection to the vast literature of geometric graph neural networks. The Pairformer module is similar to the Evoformer module of AlphaFold 2<d-cite key="jumper_highly_2021" />. In a nutshell, the Pairformer uses only scalar (SO(3) invariant) features and is an instance of an invariant geometric graph neural network (GNN). I'll then dive into geometric GNNs going beyond invariant features.*

If you are only interested in a deep dive in the _how_ of AlphaFold 3, I would advise you to take a look at a terrific blog post by Elana Simon<d-cite key="simon_illustrated_alphafold" />. If you want an in depth discussion on geometric graph neural networks, I can strongly recommend a good survey by Duval et al.<d-cite key="duval_hitchhikers_2023" />. Throughout this post I'll try to stay as close as possible to the notation used by the aformentioned survey, which is also the notation used by PyTorch geometric if `flow="target_to_source"` is used for `propagate()`.

Most output function we want to learn for chemical molecules including proteins and nucleic acids are actually *invariant* to SE(3), which are roto-translations in 3D space. Examples of such output functions are:
- Energy associated with a conformer or ionization state of a molcule.
- Forces for MD simulations are equivariant, but can be calculated from the energy by applying the gradient operator to atom positions.
- Quantum chemical properties like HOMO-LUMO gap, internal energy, or dipole moment from latent partial charges.
- Protein folding, multimer-cofolding and molecular docking. The goal is find the relative positions of atoms w.r.t each other, which is the distance matrix. The distance matrix is invariant.

This same is partially true for point clouds from lidar sensors in automated driving. We could identify relevant objects in a point cloud by just classifying the type each point belongs to, which is invariant. However, clustering of points in concepts/objects along concept hierarchies are at some point not invariant any more, but those objects have a pose <d-cite key="hinton2023represent" />. The same is true for sure for e.g. protein folding, where secondary structure (e.g. $\alpha$-helix) has actually a pose. But maybe more thoughts on this in a separate post. 

**So for the rest of this post we assume that our prediction target is invariant to roto-translations in SE(3)**. Furthermore, the prediction targets are permutation invariant with respect to the order in which the atoms are listed.

## Invariant Geometric GNNs

As the target prediction should be invariant to roto-translations, an appealing idea is to enforce this physical symmetry in the implemented graph neural networks, which leads to invariant geometric graph neural networks (GNNs). This lead to the development of powerful models like SchNet<d-cite key="schutt_schnet_2017"/>, DimeNet <d-cite key="gasteiger2020directional"/>, or GemNet<d-cite key="gasteiger2021gemnet"/>. In this section we'll explore this type of GNNs and also their limitations.

<div class="geometric_graph_invariant">
  {% include figure.liquid path="assets/img/2025-01-09_geometric-gnns_pt1/geometric_graph_invariant.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig1" class="caption" style="text-align: left;">Fig. 1 Geometric graph with invariant features. Under rotation and translation the node features $\mathbf{s}_i$ are invariant, and the node positions $\vec{x}_i$ are rotated and translated.</div>
</div>

Invariant graph neural networks (GNNs) are described by a geometric graph $$\mathcal{G} = (\mathbf{S}, \mathbf{A}, \vec{\mathbf{x}})$$ with nodes $$\mathcal{V}_{\mathcal{G}} = \{1, \ldots, n\}$$, node scalar feature matrix $$\mathbf{S} \in \mathbb{R}^{n\mathrm{x}c}$$, adjacency matrix $$\mathbf{A} \in \mathbb{R}^{n\mathrm{x}n}$$, and atom positions $$\vec{\mathbf{x}} \in \mathbb{R}^{n\mathrm{x}3}$$. Each node $$i$$ has node features $$\mathbf{s}_{i} \in \mathbb{R}^{c}$$ and a position $$\vec{x}_{i} \in \mathbb{R}^{3}$$.  The features $$\mathbf{s}_{i}$$ are invariant under transformations in SE(3), which are roto-translations in 3D (see Fig. 1). The adjacency matrix $$\mathbf{A}$$ defines the neighborhood $$\mathcal{N}_i = \{ j \in \mathcal{V}_{\mathcal{G}}\setminus i  \vert a_{ij} \neq 0 \}$$.

<div class="gnn_overview">
  {% include figure.liquid path="assets/img/2025-01-09_geometric-gnns_pt1/gnn_overview.svg" class="img-fluid rounded z-depth-1" %}
  <div id="fig2" class="caption" style="text-align: left;">Fig. 2 General network architecture of geometric GNNs.</div>
</div>

*Message passing with 2-body messages*. The core of the GNN architecture is the interaction block, which updates messages $$\mathbf{m}_{ij}$$ and then uses the updated messages from neighbors $$j \in \mathcal{N}_i$$ to update node $$i$$. The messages are updated based on invariant information like node scalars $$\mathbf{s}_i, \mathbf{s}_j$$ and invariants calculated from the node positions $$\vec{x}_i, \vec{x}_j$$. E.g. one of the early architectures (SchNet <d-cite key="schutt_schnet_2017"/>) used radial basis function embeddings based on the distance between nodes.  $$d_{ij}=\|\vec{x}_{ij}\|=\|\vec{x}_i - \vec{x}_j\|$$. The interaction block performs the following calculations:

$$\begin{align}
\mathbf{m}^{t+1}_{ij} &= msg\_upd(\mathbf{s}^t_i, \mathbf{s}^t_j, d_{ij}, \mathbf{m}_{ij}^t) \label{eq:2bodymes}\\
\mathbf{s}^{t+1}_i &= node\_upd(\mathbf{s}_{i}^{t}, \bigoplus_{j \in \mathcal{N}_i} \mathbf{m}^{t+1}_{ij}) \label{eq:2bodyupd}\,
\end{align}$$

where $$msg\_upd, node\_upd$$ are non-linear functions and $$\bigoplus$$ is a permutation invariant aggregation, because the update should not depend on the order in which we process the neighbors. The choice of $$\bigoplus$$ is important as shown by Corso et al.<d-cite key="corso2020principal"/>, but for simplicity we will use $$\sum$$ in the remainder and keep in mind that this might limit expressivity. The initial node features $$\mathbf{s}_i^0$$ are based on a learned embedding of input atom features $\mathbf{z}_i$. The learnt initial node features $$\mathbf{s}_i^0, \mathbf{s}_j^0$$, are then combined to initial messages $$\mathbf{m}_{ij}^0$$ together with the distance embeddings.

One can quickly find simple counter examples <d-cite key="joshi_expressive_2023,pozdnyakov2022incompleteness"/>, which show that two graphs with different invariant properties (e.g. area of enclosing bounding box) are not distinguishable by the simple messaging scheme of SchNet. The messages $$\mathbf{m}_{ij}$$ in Eq. \ref{eq:2bodymes} are called 2-body or 1-hop messages, as these include information (distances) of two nodes or "bodies", which is a hop over 1 edge. One can implement more powerful networks using the distances and angles between k-bodies, e.g. for 3-bodies (2-hop neighbors) using the angles $$\angle ijk = \angle (\vec{x}_{ij}, \vec{x}_{ik}) = \langle \frac{\vec{x}_{ij}}{d_{ij}} , \frac{\vec{x}_{ik}}{d_{ik}} \rangle$$. This can be further extended as shown in Fig. 3. Usually, this is not extended beyond 4-bodies, due to the increasing computational costs.

<div class="k-body">
  {% include figure.liquid loading="eager" path="assets/img/2025-01-09_geometric-gnns_pt1/multi_body_gnn.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig3" class="caption" style="text-align: left;">Fig. 3 Variants of multi-body GNNs, where each variant also uses the information from the previous one: a) 2-body using only distances <d-cite key="schutt_schnet_2017"/>; b) 3-body using distances and angles between 3 bodies <d-cite key="gasteiger2020directional,gasteiger2021gemnet"/>; c) 4-body using distances, angles between 3 bodies and dihedral angles <d-cite key="gasteiger2021gemnet"/>. The uppper row visualizes visualizes the information used to calculate $m_{ij}^{t+1}$, where we only show the additional information used. The second and third row shows two graphs $\mathcal{G}_1, \mathcal{G}_2$, which cannot be distinguisghed with the message information used in this column (examples from Joshi et al.<d-cite key="joshi_expressive_2023"/>).</div>
</div>

*Going beyond 2-body messages*. In the following equations, I slightly diverge from the presentation in Duval et al.<d-cite key="duval_hitchhikers_2023" />, because the original publications by Gasteiger et al.<d-cite key="gasteiger2020directional,gasteiger2021gemnet"/> also use the previous layer messages $$\mathbf{m}_{ij}^{t}$$. Actually the previous messages are passed through an MLP, but I leave that out for simplicity. When comparing the equations with the original publications, please note that indexing differs. In the paper the `flow="source_to_target"` notation is used: $$\mathbf{m}_{ji}$$ for messages towards node $i$. The SchNet equations are actually not the actual SchNet, but an implementation that uses the same information, but is more similar to GemNet-T/Q. All invariant GNNs presented in Fig. 3 use same node and edge updates and only differ in calculation of $$\hat{\mathbf{m}}_{ij}$$:
$$\begin{align}
\mathbf{s}^{t+1}_i    &= \mathbf{s}_{i}^{t} +  \sum_{j \in \mathcal{N}_i} f_{2-body}(d_{ij}) \odot \hat{\mathbf{m}}_{ij} \label{eq:node_upd} \\
\mathbf{m}^{t+1}_{ij} &= \hat{\mathbf{m}}_{ij} + g(\hat{\mathbf{m}}_{ij},\mathbf{s}_{i}^{t+1}, \mathbf{s}_{j}^{t+1} ) \label{eq:edge_upd}
\end{align}$$

* a) SchNet <d-cite key="schutt_schnet_2017"/> like:

$$\begin{equation}
\hat{\mathbf{m}}_{ij} = \mathbf{m}^{t}_{ij} \\
\end{equation}$$

* b) GemNet-T <d-cite key="gasteiger2021gemnet"/>:

$$\begin{equation}
\hat{\mathbf{m}}_{ij} = \mathbf{m}^{t}_{ij} + \sum_{k \in \mathcal{N}_i \setminus j} f_{3-body}(d_{ik}, \angle ijk )  \odot a(\mathbf{m}^{t}_{ik}) = \mathbf{m}^{t}_{ij} + \mathbf{m}_{ij}^{triplet} \label{eq:msg_gemnet_t}
\end{equation}$$

* c) GemNet-Q <d-cite key="gasteiger2021gemnet"/>

$$\begin{equation}\label{eq:msg_gemnet_q} 
\begin{split}
\hat{\mathbf{m}}_{ij} = \mathbf{m}^{t}_{ij} & + \mathbf{m}_{ij}^{triplet} + \\
 \sum_{\substack{m \in \mathcal{N}_i \setminus j\\n \in \mathcal{N}_m \setminus i,j\\ }} [ & f_{4-body}(d_{ij},\angle ijm , \angle ijmn)^T \mathbf{W} f_{3-body}(d_{im}, \angle imn)  \\
  & \odot f_{2-body}(d_{mn}) \\
  & \odot \mathbf{m}^t_{mn} ]
\end{split}
\end{equation}$$

As one can clearly see the message interaction calculating $$\hat{\mathbf{m}}_{ij}$$ for GemNet-Q in Eq. \ref{eq:msg_gemnet_q} is quadratic in the average number of neighbors and each calculation is more expensive as well. The approach of using only invariant information (distances & angles) is quickly computationally infeasible. Furthermore, the limitations to distinguish graphs $\mathcal{G}_1, \mathcal{G}_2$ (see Fig. 3) in the 4-body case might not be too limiting for small molecules, but they might pose a significant problem for a learning task involving longer protein chains. I want to describe to ways to mitigate this problem and then explore both of them step by step:

1. Use a fully connected graph and ...?
2. Use more informative features, e.g. equivariant cartesian tensors as node/edge information, resulting in equivariant GNN <d-cite key="schutt_equivariant_2021,wang_enhancing_2024" /> (more on that in a second post)

## Fully Connected Graphs and Transformers

So let's explore the first idea, using fully connected graphs, which all of the universal approximation theorems<d-cite key="duval_hitchhikers_2023,dym2020universality"/> use as an architecture in their proofs by aggregating over all nodes. Usually also spherical harmonics need to be included, as they are a orthogonal basis of all functions defined on the sphere. But let's not delve into that and focus on full connectivity.

*Is a fully connected graph enough?* Fig. 4 shows the distance matrices of a counter-example by Pozdnyakov et al.<d-cite key="pozdnyakov2022incompleteness"/>. We can swap columns and rows as the distances do not depend on the order in which we list the points (Remember our prediction target is permutation invariant w.r.t how we list the atoms). However, we cannot convert the matrix of $A^-$ to $A^+$ by row/column permutations due to the area marked by the red box, so the matrices are *different*. A 2-body GNN (e.g. SchNet) on a fully connected graph means passing information using the distances only within a row of the distance matrix. The aggregation operation (Eq. \ref{eq:node_upd}) then allows us to do swapping of columns for each row separately. This means we cannot distinguish the first two rows, and therefore we cannot distinguish $A^+$ and $A^-$, although they are distinct point clouds. This can cause significant problems in learning, which not only affects the counter-example due to the smoothness of predictions. Fig 4. shows some intuition on the problem. The non-distiniguishable structures are forced to have the same predicted value $\tilde{y}_{ML}$. As ML predictors are usually to some extend smooth, the neighboring structures on the data manifold are affected as well.

<div class="l-gutter">
{% include figure.liquid loading="eager" path="assets/img/2025-01-09_geometric-gnns_pt1/Pozdnyakov_2022_fig4_distance_matrix.png" class="img-fluid rounded z-depth-1" zoomable=true %} 
<div id="fig4" class="caption" style="text-align: left;">Fig 4. Top: Distance matrices of counter-example by Pozdnyakov et al.<d-cite key="pozdnyakov2022incompleteness"/>. Differences are highlighted by red box. Bottom: Visualization why non-distinguishable structures $A^+,A^-$ can pose a problem in ML.</div></div>

*What is a good criterion?* The distance matrix uniquely identifies a point cloud up to roto-translations and reflections<d-cite key="dokmanic2015euclidean"/>. This property is due to fact that one can calculate the Gram matrix $\mathbf{G}$ from the squared distance matrix via $$G_{ij} = \langle \vec{x}_i, \vec{x}_j \rangle = \frac{1}{2} ( d_{ik}^2 + d_{jk}^2 - d_{ij}^2  )$$ by choosing some point $k$ as reference. We have to choose some reference point, because the distance matrix is invariant to translations, weheras the Gram matrix is not. Via singular value decomposition can uniquely (up to rotation and reflection) derive the point positions $\vec{\mathbf{x}}$ from the Gram Matrix. So if we do not want to cover enantiomers and can assume atom properties to be invariant under SO(3) for our target function, the full distance matrix sounds like a good starting point. If our model can distinguish differences in distance matrices (and with it gram matrices), then we should be good to go. 

*How to create such models?* Lemma 1 of Villar et al. <d-cite key="villar2021scalars"/> shows that a function based on the gram matrix and node features $$f(\mathbf{G}, \mathbf{S})$$ does the job and show an example with an MLP. However, the MLP does not help, because we want to learn a function that generalizes over different input sizes and is invariant to permutations retraining. The counter-example is also distinguishable by the 3-body networks described in Eq. \ref{eq:msg_gemnet-t}. Including a fully connected graph, they are also able to distinguish the graphs $\mathcal{G}_1, \mathcal{G}_2$ of Fig. 3. While there might exist also counter-examples for these models, 3-body invariant GNNs on fully-connected graph are powerful. This means we use Eq. \ref{eq:node_upd}. \ref{eq:edge_upd}, and \ref{eq:msg_gemnet_t}, but the neighborhood $\mathcal{N}_i$ includes all other nodes. Of course, one can do better than using a simple sum in Eq. \ref{eq:node_upd} and use an adaptable aggregation by switching to transformers. This is done e.g. in Alfafold's Pairformer module.

### Alfafold 3 Pairformer

<div class="l-body">
  {% include figure.liquid path="assets/img/2025-01-09_geometric-gnns_pt1/alphafold3_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig1" class="caption" style="text-align: left;">Fig. 1 Structural overview from AlphaFold 3<d-cite key="abramson_accurate_2024" />.</div>
</div>

The input to the Pairformer are the representations for pair $$\mathbf{m}_{ij} \in \mathbb{R}^{c_{pair}}$$ and single $$\mathbf{s}_{i} \in \mathbb{R}^{c_{single}}$$ with $$i,j \in \{1, ..., N_{Tokens}\}$$. Here I already explicitly used the symbols $$\mathbf{m}_{ij}$$ and $$\mathbf{s}_{i}$$ to stay in the GNN jargon for messages and node features, respectively. Each token can represent different types of entities (whole amino acid, whole nucleotide, single atom) but for our discussion this is mainly irrelevant. We could think of each token representing a single atom in a molecule. The pair represenation also contains informations from templates and an mutliple sequence alignment (MSA), but AlphaFold3 can be run without, so let's assume we do that.

The input representations $$\mathbf{m}_{ij}, \mathbf{s}_{i}$$ are created from the input embedding $$\mathbf{s_i^{input}}$$. $$\mathbf{s_i^{input}}$$ is calculated in the "Input imbedder" block based on network inputs. Leaving details like sparse local attention and distances only available within the generated conformer (see Supplements of <d-cite key="abramson_accurate_2024" />) out, we end up with SO(3) invariant features that contain information on atom features and on pairwise distances between atoms within one conformer. The input representations are created as 

$$
\begin{aligned}
\mathbf{m}_{ij} &= f_m(\mathbf{s_i^{input}}) + f_m(\mathbf{s_j^{input}}) + rpe(i,j) \\
\mathbf{s}_{i}  &= f_s(\mathbf{s_i^{input}})
\end{aligned}
$$ 

where $$rpe(i,j)$$ is the relative positional encoding. *Interestingly, the $rpe$ encodes (beside other parts)the token index distance (L. 6 Algorithm 3 Supplement <d-cite key="abramson_accurate_2024" />), which makes sense for protein sequences, but not for the atoms of the ligand, as it breaks the permutation invariance*. The distance of tokens now depends how we read in the atoms of the ligand at the network input.

<div class="l-body">
  {% include figure.liquid path="assets/img/2025-01-09_geometric-gnns_pt1/alphafold3_pairformer_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig2" class="caption" style="text-align: left;">Fig. 2 Pairformer module overview from AlphaFold 3<d-cite key="abramson_accurate_2024" />.</div>
</div>

The main action on the pair representation within Pairformer are the triangle updates and triangle self-attention. A lot is already explained on the motivation of these updates in Elana's post<d-cite key="simon_illustrated_alphafold" />, so I just want to emphasize here that it does not enforce the triangle inequality but it is only an inductive bias. The modules perform the following updates, where functions $a, b, g, q, k \ldots$ can be different for each module and we is always perform $$\mathbf{m}_{ij}' = \mathbf{m}_{ij} + \tilde{\mathbf{m}}_{ij}$$:

* Triangular update "outgoing" edges: $$\begin{equation}\tilde{\mathbf{m}}_{ij} = g(\mathbf{m}_{ij}) \odot \sum_k a(\mathbf{m}_{ik}) \odot b(\mathbf{m}_{jk})\end{equation}$$

* Triangular update "incoming" edges: $$\begin{equation}\tilde{\mathbf{m}}_{ij} = g(\mathbf{m}_{ij}) \odot \sum_k a(\mathbf{m}_{ki}) \odot b(\mathbf{m}_{kj})\end{equation}$$

* Triangular attention starting node: 
$$\begin{equation}
\begin{split}
\alpha_{ijk} &= softmax_k(\frac{1}{\sqrt{c}}q(\mathbf{m}_{ij})^T k(\mathbf{m}_{ik}) + b(\mathbf{m}_{jk})) \\
\tilde{\mathbf{m}}_{ij} &= g(\mathbf{m}_{ij}) \odot \sum_k a_{ijk} \, v(\mathbf{m}_{ik}) 
\end{split}
\end{equation}$$

* Triangular attention ending node:
$$\begin{equation}
\begin{split}
\alpha_{ijk} &= softmax_k(\frac{1}{\sqrt{c}}q(\mathbf{m}_{ij})^T k(\mathbf{m}_{kj}) + b(\mathbf{m}_{ki})) \\
\tilde{\mathbf{m}}_{ij} &= g(\mathbf{m}_{ij}) \odot \sum_k a_{ijk} \, v(\mathbf{m}_{kj}) 
\end{split}
\end{equation}$$

The above attention updates ignore the fact that we have multiple attention heads, but this only affects the channel dimension. The main action on the single representation is the single attention with pair bias:

$$\begin{align}
\alpha_{ij} &= softmax_j(\frac{1}{\sqrt{c}}q(\mathbf{s}_i^t)^T k(\mathbf{s}_j^t)+b(\mathbf{m}_{ij}))  \\
\mathbf{s}_i^{t+1} &= \mathbf{s}_i^t + g(\mathbf{s}_i^t) \odot \sum_j \alpha_{ij} \, v(\mathbf{s}_j^t)
\end{align}$$

where the functions $q, k, v$ are the query, key, and value vectors of transformers, $b$ is the bias function based on the pairs, and $g$ is a gating function as before for the pairs.


##

So let's rewrite the 3-body variant GemNet-T (Eq. \ref{eq:msg_gemnet-t}) for a fully connected graph and switch to using only distances due to the law of cosines. Furthermore, instead of multiplying the distance and angle embeddings in every message update, we could also simply use the respective messages, as we already encoded the distances in the embedding of the messages:

$$\begin{equation}\begin{split}
\hat{\mathbf{m}}_{ij} &= \mathbf{m}^{t}_{ij} + \sum_{k} f_{3-body}(d_{ik}, \angle ijk )  \odot a(\mathbf{m}^{t}_{ik}) \\
&= \mathbf{m}^{t}_{ij} + \sum_{k} f_{3-body}(d_{ij}, d_{ik}, d_{jk}) \odot a(\mathbf{m}^{t}_{ik}) \qquad \text{#Law of cosines}\\
& =\mathbf{m}^{t}_{ij} + \sum_{k} a(\mathbf{m}^{t}_{ik}) \odot g(\mathbf{m}^{t}_{ij}) \odot b(\mathbf{m}^{t}_{jk}) \qquad \text{#Distances embedded in messages}\\ 
& =\mathbf{m}^{t}_{ij} + g(\mathbf{m}^{t}_{ij}) \odot \sum_{k} a(\mathbf{m}^{t}_{ik}) \odot b(\mathbf{m}^{t}_{jk}) \label{eq:msg_gemnet-t-distances}
\end{split}\end{equation}$$ 

Furthermore, the actual 
