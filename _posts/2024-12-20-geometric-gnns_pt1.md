---
layout: distill
title: Alpha Fold 3 and Geometric Graph Neural Networks
description: A small journey from Pairformer (AlphaFold), over geometric vector percetrons (GVPs), to ViSNet for molecular graphs.
draft: true
tags:
giscus_comments: false
date: 2024-12-29
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
    url: "https://drbaer.github.io"
output: distill::distill_article
bibliography: 2024-12-20-geometric-gnns_pt1.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Background
  - name: AlphaFold 3 Pairformer
  - name: Classes of Equivariant Graph Neural Networks (GNNs)

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
In this post I am exploring the representation learning module (Pairformer) of AlphaFold 3<d-cite key="abramson_accurate_2024" /> and it's connection to the vast literature of geometric graph neural networks. The Pairformer module is similar to the Evoformer module of AlphaFold 2<d-cite key="jumper_highly_2021" />. In a nutshell, the Pairformer uses only scalar (SO(3) invariant) features and is an instance of an invariant geometric graph neural network (GNN). I'll then dive into geometric GNNs going beyond invariant features.

If you are only interested in a deep dive in the _how_ of AlphaFold 3, I would advise you to take a look at a terrific blog post by Elana Simon<d-cite key="simon_illustrated_alphafold" />. If you want an in depth discussion on geometric graph neural networks, I can strongly recommend a good survey by Duval et al.<d-cite key="duval_hitchhikers_2023" />. Throughout this post I'll try to stay as close as possible to the notation used by the aformentioned survey, which is also the notation used by PyTorch geometric if `flow="target_to_source"` is used for `propagate()`.

## Background

Most outputs for chemical learning molecules are actually invariant to SE3. This includes:
- includes Energy and forces (derivative of energy), e.g. for MD simulations
- Quantum physics based methods like HOMO, LUMO, ... (see QM9)
- Protein folding and molecular docking (only relative positions of atoms are relevant (distances + angles))

Same is usually true for for point clouds from lidar sensors in automated driving with the exceptoin 

### Invariant Geometric GNNs

<div class="2-body">
  {% include figure.liquid path="assets/img/2024-12-20-geometric-gnns_pt1/geometric_graph_invariant.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig1" class="caption" style="text-align: left;">Fig. 1 Geometric graph with invariant features. Under rotation and translation the node features $\mathbf{s}_i$ are invariant, and the node positions $x_i$ are rotated and translated.</div>
</div>

Invariant graph neural networks (GNNs) are described by a geometric graph $$\mathcal{G} = (\mathbf{S}, \mathbf{A}, \mathbf{x})$$ with nodes $$\mathcal{V}_{\mathcal{G}} = \{1, \ldots, N\}$$. Each node $$i$$ has node features $$\mathbf{s}_{i} \in \mathbb{R}^{c}$$ and a position $$\vec{x}_{i} \in \mathbb{R}^{3}$$. The features $$s_{i}$$ are invariant under transformations in SE(3), which are roto-translations in 3D. The adjacency matrix $$\mathbf{A} \in \mathbb{R}^{N\mathrm{x}N}$$ defines the neighborhood $$\mathcal{N}_i = \{ j \in \mathcal{V}_{\mathcal{G}}\setminus i  \vert a_{ij} \neq 0 \}$$.


The overall structure can be summarized 

Invariant geometric GNNs aggregate messages $$\mathbf{m}_{ij}$$ of neighbors $$j \in \mathcal{N}_i$$ to update node $$i$$. The messages are based only on invariant information like node scalars $$\mathbf{s}_i, \mathbf{s}_j$$ and invariante edge scalars $$\mathbf{e}_{ij}$$.

$$\begin{align}
\mathbf{m}^{t+1}_{ij} &= f(\mathbf{s}^t_i, \mathbf{s}^t_j, \mathbf{e}_{ij}, \mathbf{m}_{ij}^t)\\
\mathbf{s}^{t+1}_i &= g(\mathbf{s}_{i}^{t}, \bigoplus_{j \in \mathcal{N}_i} \mathbf{m}^{t+1}_{ij}),
\end{align}$$

where $$f, g$$ are non-linear functions and $$\bigoplus$$ is a permutation invariant aggregation<d-cite key="corso2020principal"/>, because the update should not depend on the order in which we process the neighbors. In one of the early invariant geometric GNNs (SchNet <d-cite key="schutt_schnet_2017"/>), the edge features are based on an embedding (e.g. radial-basis functions) of the distance between nodes: $$d_{ij}=\|\vec{x}_{ij}\|=\|\vec{x}_i - \vec{x}_j\|$$. The initial messages $$\mathbf{m}_{ij}^0$$ are created via initial atom embeddings $$\mathbf{s}_i^0, \mathbf{s}_j^0$$.


However, one can quickly find simple counter examples, which show that two graphs with different invariant properties (e.g. area of enclosing bounding box) are not distinguishable by this simple messaging scheme<d-cite key="pozdnyakov2022incompleteness,joshi_expressive_2023"/>. The messages $$\mathbf{m}_{ij}$$ are called 2-body or 1-hop messages, as these include information from two nodes or "bodies", which is a hop over 1 edge. One can implement more powerful networks using the distances and angles between k-bodies, e.g. for 3-bodies (2-hop neighbors) using the angles $$\angle ijk = \angle (\vec{x}_{ij}, \vec{x}_{ik}) = \langle \frac{\vec{x}_{ij}}{d_{ij}} , \frac{\vec{x}_{ik}}{d_{ik}} \rangle$$. This can be further extended as shown in Fig. 3.

<div class="k-body">
  {% include figure.liquid path="assets/img/2024-12-20-geometric-gnns_pt1/multi_body_gnn.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig3" class="caption" style="text-align: left;">Fig. 3 Variants of multi-body GNNs, where each variant also uses the information from the previous one: a) 2-body using only distances <d-cite key="schutt_schnet_2017"/>; b) 3-body using distances and angles between 3 bodies <d-cite key="gasteiger2020directional"/>; c) 4-body using distances, angles between 3 bodies and dihedral angles <d-cite key="gasteiger2021gemnet"/>. The uppper row visualizes visualizes the information used to calculate $m_{ij}^{t+1}$. The second and third row shows two graphs $\mathcal{G}_1, \mathcal{G}_2$, which cannot be distinguisghed with the message information used in this column (examples from Joshi et al.<d-cite key="joshi_expressive_2023"/>).</div>
</div>

In Fig. 3, I slightly diverged from the presentation in Duval et al.<d-cite key="duval_hitchhikers_2023" />, because the original publications by Gasteiger et al.<d-cite key="gasteiger2020directional,gasteiger2021gemnet"/> do not recalculate the messages from the node scalars, but only use the edge attributes $$\mathbf{e}_{ij}$$ and the previous messages $$\mathbf{m}_{ij}^{t+1}$$. When reading the original publications by Gasteiger et al. also note, that indexing differs, as they use the `flow="source_to_target"` notation: $$\mathbf{m}_{ji}$$ for messages towards node $i$. Specifically the invariant GNNs presented in Fig. 3 use the following updates:

* a) SchNet<d-cite key="schutt_schnet_2017"/>:

$$\begin{align*}
\mathbf{m}^{t+1}_{ij} &= \mathbf{s}^t_j \odot f(d_{ij}) \\
\mathbf{s}^{t+1}_i &= \mathbf{s}_{i}^{t} +  \sum_{j \in \mathcal{N}_i} \mathbf{m}^{t+1}_{ij}
\end{align*}$$

* b) DimeNet<d-cite key="gasteiger2020directional"/>:

$$\begin{align*}
\mathbf{m}^{t+1}_{ij} &= \mathbf{s}^t_j \odot f(d_{ij}) \\
\mathbf{s}^{t+1}_i &= \mathbf{s}_{i}^{t} +  \sum_{j \in \mathcal{N}_i} \mathbf{m}^{t+1}_{ij}
\end{align*}$$

* c) GemNet <d-cite key="gasteiger2021gemnet"/>

$$\begin{align*}
\mathbf{m}^{t+1}_{ij} &= \mathbf{s}^t_j \odot f(d_{ij}) \\
\mathbf{s}^{t+1}_i &= \mathbf{s}_{i}^{t} +  \sum_{j \in \mathcal{N}_i} \mathbf{m}^{t+1}_{ij}
\end{align*}$$



### Transformers are Graph Neural Networks



## AlphaFold 3 Pairformer

<div class="l-body">
  {% include figure.liquid path="assets/img/2024-12-20-geometric-gnns_pt1/alphafold3_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
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

where $$rpe(i,j)$$ is the relative positional encoding. Interestingly, the $rpe$ encodes (beside other parts)the token index distance (L. 6 Algorithm 3 Supplement <d-cite key="abramson_accurate_2024" />), which makes sense for protein sequences, but not for the atoms of the ligand, as it breaks the permutation invariance. The distance of tokens now depends how we read in the atoms of the ligand at the network input.

<div class="l-body">
  {% include figure.liquid path="assets/img/2024-12-20-geometric-gnns_pt1/alphafold3_pairformer_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  <div id="fig2" class="caption" style="text-align: left;">Fig. 2 Pairformer module overview from AlphaFold 3<d-cite key="abramson_accurate_2024" />.</div>
</div>

The main action on the pair representation within Pairformer are the triangle updates and self attention. A lot is already explained on the motivation of these updates in Elana's post<d-cite key="simon_illustrated_alphafold" />, so I just want to emphasize here that it does not enforce the triangle inequality but it is only an inductive bias. The modules perform the following updates, where functions $a, b, g, q, k \ldots$ can be different for each module and we is always perform $$\mathbf{m}_{ij}' = \mathbf{m}_{ij} + \tilde{\mathbf{m}}_{ij}$$:

* Triangular update "outgoing" edges: $$\tilde{\mathbf{m}}_{ij} = g(\mathbf{m}_{ij}) \odot \sum_k a(\mathbf{m}_{ik}) \odot b(\mathbf{m}_{jk})$$
* Triangular update "incoming" edges: $$\tilde{\mathbf{m}}_{ij} = g(\mathbf{m}_{ij}) \odot \sum_k a(\mathbf{m}_{ki}) \odot b(\mathbf{m}_{kj})$$
* Triangular attention starting node: $$\tilde{\mathbf{m}}_{ij} = g(\mathbf{m}_{ij}) \odot \sum_k f(\mathbf{m}_{ki}) \odot g(\mathbf{m}_{kj})$$
* Triangular attention ending node: $$\tilde{\mathbf{m}}_{ij} = h(\mathbf{m}_{ij}) \odot \sum_k f(\mathbf{m}_{ki}) \odot g(\mathbf{m}_{kj})$$

The above attention updates ignore the fact that we have multiple attention heads, but this only affects in the channel dimension.