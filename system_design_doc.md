# Risk Estimation and Control in Dynamic Systems
Notes by Miguel Bueno

## Abstract
We consider a population of entities $\mathcal{X}$ evolving over time $t$,
where each entity $x_i$ has a latent true label $Y_i$ and engagement
weight $E_{it}$. The target estimand is the engagement-weighted
prevalence $\mu_E = \mathbb{E}_{x \sim p_t}[Y(x)] \;\text{with}\;
p_t(x) \propto E_{it}$, and the global objective is to minimize an upper
confidence bound on a policy-dependent loss functional
$\mathcal{L}(\tau,\mathcal{G})$ at $t+1$ by jointly optimizing the
deployment policy $\tau$ and the labeling policy $\mathcal{G}$.

We have available a set of resources $\mathcal{R}$ categorized by type
$\kappa = \rho(r)$<sup>1</sup>. Each resource is characterized by a
Resource Profile $\Phi_\kappa = \langle c_\kappa, \Omega_\kappa,
\boldsymbol{N}_\kappa \rangle$, representing its marginal cost,
processing capacity, and noise<sup>2</sup><sup>3</sup>. Under a fixed
global budget $B_t$ and total system bandwidth $\Omega_t$, the policies
$\tau$ and $\mathcal{G}$ are constrained such that aggregate cost and
throughput satisfy budget and capacity constraints (e.g.,
$\sum_{\kappa} c_\kappa n_\kappa \leq B_t$ and $\sum_{\kappa}
\Omega_\kappa \leq \Omega_t$).

A resource type $\kappa$ is said to be high-fidelity if it exhibits low
noise, $\boldsymbol{N}_{\kappa} \approx \mathbf{I}$, but is constrained
in cost and/or capacity, such that $c_{\kappa}$ is large and
$\Omega_\kappa \ll |\mathcal{X}|$. Conversely, a resource type $\kappa$
is low-fidelity if it is inexpensive and scalable, with $c_\kappa \ll
c_{\mathrm{HF}}$ and $\Omega_\kappa \gtrsim |\mathcal{X}|$, but exhibits
substantially higher noise<sup>4</sup>. A resource type $\kappa$ is
mid-fidelity if it lies between these extremes in noise, cost, and
capacity, while still remaining meaningfully capacity-constrained
relative to the low-fidelity resource.

High- and mid-fidelity resources are used to produce reference labels
$Y_i^*$ for estimating $\mu_E$. We construct a sampled dataset
$\mathcal{D}_{n,t} = \{X_i \mid R_{it}=1\} \subset \mathcal{X}$ with
inclusion probabilities $\pi_{it}$. To efficiently estimate $\mu_E$,
samples are drawn under an importance distribution $q(x)$ (potentially
stratified over $\mathcal{H}$ and pooled over horizon $T$), with
importance weights $w_i = p_{it} \cdot (q_{it})^{-1}$. Each sampled
entity $X_i$ is evaluated by a panel of mid-fidelity resources
$\mathcal{P}_i \subset \mathcal{R}$ under policy $\mathcal{G}$. Each
resource $r \in \mathcal{P}_i$ produces a valuation $v_{ir}$<sup>5</sup>,
which is aggregated via $\mathcal{A}(\cdot)$ to produce an aggregated
valuation $\hat{V}_{i} = \mathcal{A}(\{v_{ir}\}_{r \in
\mathcal{P}_i})$ for all $x_{i} \in \mathcal{D}_{n,t}$. These valuations
are mapped to discrete outcomes using a decision threshold $\eta$. The
predicted label $\hat{Y}_{i} = \mathbf{1}\{\hat{V}_{i}>\eta\}$ is then
used to estimate $\mu_{E}$.

To minimize redundant expenditure, we implement adaptive task assignment
where panel size $|\mathcal{P}_i| = \phi(\alpha_i, \epsilon)$ is
dynamically scaled<sup>6</sup>. When $\mathcal{P}_i$ used to construct
$\hat{Y}_{i}$ consists of mid-fidelity resources, we occasionally
leverage high-fidelity resources with difference estimators to reduce
the variance of the estimator<sup>7</sup>.

Where appropriate, low-fidelity resources are trained, fine-tuned, and
tested using $Y_i^*$. They, in turn, generate valuations $v_{ir}$ for
$\mathcal{X}$. These valuations drive $\tau$, inducing dynamics governed
by $p_{t+1} = \mathcal{F}(p_t, \tau_t)$. After calibration via
$g(\cdot)$, these valuations inform both $\pi_{it}$ for the production of
$Y_i^*$ and the deployment policy $\tau$, which maps calibrated scores to
exposure levels $a_\tau(x) \in [0,1]$ interpreted as a multiplicative
ranking factor<sup>8</sup><sup>9</sup>.

Low-fidelity resource performance is assessed using a self-normalized,
importance-weighted confusion matrix $\boldsymbol{M}(\eta)$, yielding
weighted precision $p_{\omega m}(\boldsymbol{M})$ and recall
$r_{\omega m}(\boldsymbol{M})$ and other metrics. Evaluation is conducted
under a counterfactual no-policy baseline by weighting observations using
predicted counterfactual engagement $\widehat{E}_{it}$ to avoid demotion
bias. When $\mathcal{P}_i$ used to construct $Y_i^*$ consists of
mid-fidelity resources, we refer to the reference label as noisy and
denote it $\tilde{Y}_i$. To evaluate each panel, a subsample
$\mathcal{D}_{n^{(2)}} \subset \mathcal{D}_{n,t}$ is adjudicated by
high-fidelity resources. All resources are evaluated for fairness for
protected partitions of entities, $\mathcal{M} \subset \mathcal{X}$ using
the raw<sup>10</sup> false positive rate difference,
$\Delta_{\mathcal{M},\mathcal{X}}\text{FPR}$, where the baseline
represents the aggregate performance across the entire population
$\mathcal{X}$. To optimize $B_{t}$, the system utilizes adaptive sampling
schemes designed to maximize the informational utility of each acquired
label<sup>11</sup>.

We can influence resource characteristics $\Phi_{\kappa}$ through
treatments $W$<sup>12</sup> whose effects are often estimated via AB
tests<sup>13</sup> that condition on additional features $Z_{i}$.
Furthermore, we can identify resources $r$ with unusually unsatisfactory
characteristics relative to those of the same type $\kappa$ and exclude
them from $\mathcal{R}$<sup>14</sup>.

Uncertainty in all estimators $\hat{\theta}$ is quantified via a
resampling operator $\mathcal{B}$, which induces bootstrap replicates
$\mathcal{D}^{(b)}_{n,t} \sim \mathcal{B}(\mathcal{D}_{n,t})$, with
estimates $\hat{\theta}^{(b)}$ summarized through their empirical
distribution<sup>15</sup>.

Since $|\mathcal{X}|$ and most $|\mathcal{D}_{n,t}|$ are large, engines
are configured to use Poisson sampling variants to mimic multinomial
draws and facilitate high-throughput<sup>16</sup>.

Collectively, these components define a closed-loop system in which
sampling, labeling, evaluation, and deployment interact through shared
resource constraints and evolving population dynamics. The system
objective is therefore to solve the constrained optimization problem
$\min_{\tau, \mathcal{G}} \text{UCB}(\mathcal{L}(\tau,\mathcal{G}))$
subject to $\sum_{i \in \mathcal{D}_{n,t}} \sum_{r \in
\mathcal{P}_i} c_{\rho(r)} \le B_t$ and $\sum_{i \in
\mathcal{D}_{n,t}} \mathbf{1}\{\rho(r)=\kappa,\; r \in
\mathcal{P}_i\} \le \Omega_\kappa,  \forall \kappa$.

---

<sup>1</sup> E.g. deep neural-networks, large language models, novice
and expert human annotators, etc.  
<sup>2</sup> Such characteristics can evolve over time. However, we
assume they are relatively stable in the short-run.  
<sup>3</sup> In reality noise is often heteroscedastic, arising from
resource capabilities and latent task complexity.  
<sup>4</sup> As reflected by $\text{tr}(\boldsymbol{N}_\kappa) \gg
\text{tr}(\boldsymbol{N}_{\mathrm{HF}})$.  
<sup>5</sup> For human annotators, $v_{ir}$ is typically derived from
structured ordinal responses $S_{ir\ell}$ over a template
$\mathcal{Q}$.  
<sup>6</sup> Task difficulty is frequently proxied by agreement
($\alpha_i$) calculated over an initial rater subset.  
<sup>7</sup> E.g. prediction-power-inference (PPI), double-sampling, etc.  
<sup>8</sup> $a_\tau(x)=1$ corresponds to no demotion and $a_\tau(x)=0$
corresponds to complete filtering.  
<sup>9</sup> Where explainability is required, $g(v_{ir})$ may be
accompanied by a vector of interpretive features.  
<sup>10</sup> Non-engagement weighted.  
<sup>11</sup> E.g. multiple importance sampling (MIS), adaptive sampling,
and sequential stopping rules.  
<sup>12</sup> E.g. changes to a template $\mathcal{Q}$, refinements to
the labeling policy $\mathcal{G}$, etc.  
<sup>13</sup> i.e randomized control trials.  
<sup>14</sup> When $r$ corresponds to a human annotator, such
interventions must comply with applicable co-employment laws.  
<sup>15</sup> $\mathcal{B}$ typically leverages a paired pigeonhole
bootstrapping but can leverage other bootstraps.  
<sup>16</sup> These are co-deployed with Gumbel-Max-style weighted
reservoir sampling procedures.

## Glossary

### Population and Target

$\mathcal{X}$ — Population of entities  
$x, x_i$ — Entity in the population  
$i$ — Entity index  
$t$ — Time index  
$Y_i$ — Latent true label (1 = positive class)  
$E_{it}$ — Engagement weight of entity $x_i$ at time $t$  
$p_t(x)$ — Nominal (traffic-weighted) distribution, $p_t(x) \propto E_{it}$  
$\mu_E = \mathbb{E}_{x \sim p_t}[Y(x)]$ — Engagement-weighted prevalence  

### Sampling

$X_i$ — Sampled entity  
$R_{it}$ — Sampling indicator  
$\pi_{it} = \mathbb{P}(R_{it}=1)$ — Inclusion probability  
$\mathcal{D}_{n,t}$ — Sampled dataset $\{X_i \mid R_{it}=1\}$  
$q(x)$ — Importance sampling distribution  
$w_i = \frac{p_t(X_i)}{q(X_i)}$ — Importance weight  
$\mathcal{H}$ — Set of strata  
$h$ — Stratum index  
$T$ — Time horizon for pooled sampling  

### Resources and Policies

$\mathcal{R}$ — Set of resources  
$r$ — Resource index  
$\mathcal{K}$ — Set of resource types  
$\kappa$ — Resource type index  
$\rho(r)$ — Mapping from resource to type  
$\Phi_\kappa = \langle c_\kappa, \Omega_\kappa, \mathbf{N}_\kappa \rangle$ — Resource profile  
$c_\kappa$ — Marginal cost  
$\Omega_\kappa$ — Processing capacity  
$\mathbf{N}_\kappa$ — Noise structure  
$B_t$ — Total budget  
$\mathcal{G}$ — Labeling policy  

### Annotation and Valuation

$\mathcal{P}_i \subset \mathcal{R}$ — Panel of resources  
$v_{ir}$ — Valuation  
$\mathcal{A}(\cdot)$ — Aggregation function  
$\hat{V}_i = \mathcal{A}(\{v_{ir}\})$ — Aggregated valuation  
$\eta$ — Decision threshold  
$\hat{Y}_i = \mathbb{I}(\hat{V}_i > \eta)$ — Predicted label  
$\tilde{Y}_i$ — Noisy reference label  
$\mathcal{D}_{n^{(2)}}$ — High-fidelity subsample  

### Structured Human Annotation

$\mathcal{R}^{(H)}$ — Human annotators  
$\mathcal{Q}$ — Annotation questions  
$\ell$ — Question index  
$S_{ir\ell}$ — Ordinal response  
$v_{ir}$ — Derived valuation  

### Prediction and Deployment

$v(x)$ — Scalable valuation  
$g(\cdot)$ — Calibration  
$a_\tau(x) \in [0,1]$ — Exposure multiplier  
$\tau$ — Deployment policy  
$p_{t+1} = \mathcal{F}(p_t, \tau_t)$ — Population dynamics  

### Evaluation

$\mathbf{M}(\eta)$ — Weighted confusion matrix  
$p_{\omega}(\mathbf{M})$ — Weighted precision  
$r_{\omega}(\mathbf{M})$ — Weighted recall  
$\hat{E}_{it}$ — Counterfactual engagement  
$\mathcal{M}$ — Protected subset  
$\Delta_{\mathcal{M},\mathcal{X}}\mathrm{FPR}$ — FPR difference  

### Objective and Optimization

$\mathcal{L}(\tau,\mathcal{G})$ — Loss functional  
$\operatorname{UCB}(\cdot)$ — Upper confidence bound  
$\min_{\tau,\mathcal{G}} \operatorname{UCB}(\mathcal{L})$ — System objective  
$n_\kappa$ — Assignments per type  

### Uncertainty

$\hat{\theta}$ — Estimator  
$\mathcal{B}$ — Bootstrap operator  
$b$ — Bootstrap index  
$\mathcal{D}^{(b)}_{n,t}$ — Resampled dataset  
$\hat{\theta}^{(b)}$ — Bootstrap estimate  

### Causal Variables

$W$ — Treatment  
$Z_i$ — Covariates  
