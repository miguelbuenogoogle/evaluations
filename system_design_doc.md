# Risk Estimation and Control in Dynamic Systems
Notes by Miguel Bueno

## Abstract

We consider a population of entities $\mathcal{X}$ evolving over time $t$,
where each entity $x\_i$ has a latent true label $Y\_i$ and engagement
weight $E\_{it}$. The target estimand is the engagement-weighted
prevalence $\mu\_E = \mathbb{E}\_{x \sim p\_t}[Y(x)] \mid p\_t(x) \propto
E\_{it}$,and the global objective is to minimize an upper confidence
bound on a policy-dependent loss functional $\mathcal{L}(\tau,\mathcal{G})$
at $t+1$ by jointly optimizing the deployment policy $\tau$ and the
labeling policy $\mathcal{G}$.

We have available a set of resources $\mathcal{R}$ categorized by type
$\kappa = \rho(r)$<sup>1</sup>. Each resource is characterized by a
Resource Profile $\Phi\_\kappa = \langle c\_\kappa, \Omega\_\kappa,
\mathbf{N}\_\kappa \rangle$, representing its marginal cost, processing
capacity, and noise<sup>2</sup><sup>3</sup>. Under a fixed global budget
$B\_t$ and total system bandwidth $\Omega\_t$, the policies $\tau$ and
$\mathcal{G}$ are constrained such that aggregate cost and throughput
satisfy budget and capacity constraints (e.g., $\sum\_{\kappa} c\_\kappa
n\_\kappa \leq B\_t$ and $\sum\_{\kappa} \Omega\_\kappa \leq \Omega\_t$).

A resource type $\kappa$ is said to be high-fidelity if it exhibits low
noise, $\mathbf{N}\_{\kappa} \approx \mathbf{I}$, but is constrained in
cost and/or capacity, such that $c\_{\kappa}$ is large and $\Omega\_\kappa
\ll |\mathcal{X}|$. Conversely, a resource type $\kappa$ is low-fidelity
if it is inexpensive and scalable, with $c\_\kappa \ll c\_{\mathrm{HF}}$
and $\Omega\_\kappa \gtrsim |\mathcal{X}|$, but exhibits substantially
higher noise<sup>4</sup>. A resource type $\kappa$ is mid-fidelity if it
lies between these extremes in noise, cost, and capacity, while still
remaining meaningfully capacity-constrained relative to the low-fidelity
resource.

High- and mid-fidelity resources are used to produce reference labels
$Y\_i^*$ for estimating $\mu\_E$. We construct a sampled dataset
$\mathcal{D}\_{n,t} = \{X\_i \mid R\_{it}=1\} \subset \mathcal{X}$ with
inclusion probabilities $\pi\_{it}$. To efficiently estimate $\mu\_E$,
samples are drawn under an importance distribution $q(x)$ (potentially
stratified over $\mathcal{H}$ and pooled over horizon $T$), with
importance weights $w\_i = p\_{it} \cdot (q\_{it})^{-1}$. Accounting for
an availability probability $P(A_{i}=1|Y_{i}=1) \leq 1$ at the labeling stage,
each available sampled entity $X\_i$ is evaluated by a panel of mid-fidelity
resources $\mathcal{P}\_i \subset \mathcal{R}$ under policy $\mathcal{G}$. Each
resource $r \in \mathcal{P}\_i$ produces a valuation $v\_{ir}$<sup>5</sup>,
which is aggregated via $\mathcal{A}(\cdot)$ to produce an aggregated
valuation $\hat{V}\_{i} = \mathcal{A}(\{v\_{ir}\}\_{r \in \mathcal{P}\_i})$
for all $x\_{i} \in \mathcal{D}\_{n,t}$. These valuations are mapped to
discrete outcomes using a decision threshold $\eta$. The predicted label
$\hat{Y}\_{i} = \mathbb{I}(\hat{V}\_{i}>\eta)$ is then used to estimate
$\mu\_{E}$.

To minimize redundant expenditure, we implement adaptive task assignment
where panel size $|\mathcal{P}\_i| = \phi(\alpha\_i, \epsilon)$ is
dynamically scaled<sup>6</sup>. When $\mathcal{P}\_i$ used to construct
$\hat{Y}\_{i}$ consists of mid-fidelity resources, we occasionally
leverage high-fidelity resources with difference estimators to reduce
the variance of the estimator<sup>7</sup>.

Where appropriate, low-fidelity resources are trained, fine-tuned, and
tested using $Y\_{i}^{\*}$. They, in turn, generate valuations $v\_{ir}$ for
$\mathcal{X}$. These valuations drive $\tau$, inducing dynamics governed
by $p\_{t+1} = \mathcal{F}(p\_t, \tau\_t)$. After calibration via
$g(\cdot)$, these valuations inform both $\pi\_{it}$ for the production of
$Y\_i^*$ and the deployment policy $\tau$, which maps calibrated scores to
exposure levels $a\_\tau(x) \in [0,1]$ interpreted as a multiplicative
ranking factor<sup>8</sup><sup>9</sup>.

Low-fidelity resource performance is assessed using a self-normalized,
importance-weighted confusion matrix $\mathbf{M}(\eta)$, yielding
weighted precision $p\_{\omega m}(\mathbf{M})$ and recall
$r\_{\omega m}(\mathbf{M})$ and other metrics. Evaluation is conducted
under a counterfactual no-policy baseline by weighting observations using
predicted counterfactual engagement $\widehat{E}\_{it}$ to avoid demotion
bias. When $\mathcal{P}\_i$ used to construct $Y\_i^*$ consists of
mid-fidelity resources, we refer to the reference label as noisy and
denote it $\tilde{Y}\_i$. To evaluate each panel, a subsample
$\mathcal{D}\_{n^{(2)}} \subset \mathcal{D}\_{n,t}$ is adjudicated by
high-fidelity resources. All resources are evaluated for fairness for
protected partitions of entities, $\mathcal{M} \subset \mathcal{X}$ using
the raw<sup>10</sup> false positive rate difference,
$\Delta\_{\mathcal{M},\mathcal{X}}\text{FPR}$, where the baseline
represents the aggregate performance across the entire population
$\mathcal{X}$. To optimize $B\_{t}$, the system utilizes adaptive sampling
schemes designed to maximize the informational utility of each acquired
label<sup>11</sup>.

We can influence resource characteristics $\Phi\_{\kappa}$ through
treatments $W$<sup>12</sup> whose effects are often estimated via AB
tests<sup>13</sup> that condition on additional features $Z\_{i}$.
Furthermore, we can identify resources $r$ with unusually unsatisfactory
characteristics relative to those of the same type $\kappa$ and exclude
them from $\mathcal{R}$<sup>14</sup>.

Uncertainty in all estimators $\hat{\theta}$ is quantified via a
resampling operator $\mathcal{B}$, which induces bootstrap replicates
$\mathcal{D}^{(b)}\_{n,t} \sim \mathcal{B}(\mathcal{D}\_{n,t})$, with
estimates $\hat{\theta}^{(b)}$ summarized through their empirical
distribution<sup>15</sup>.

Since $|\mathcal{X}|$ and most $|\mathcal{D}\_{n,t}|$ are large, engines
are configured to use Poisson sampling variants to mimic multinomial
draws and facilitate high-throughput<sup>16</sup>.

Collectively, these components define a closed-loop system in which
sampling, labeling, evaluation, and deployment interact through shared
resource constraints and evolving population dynamics. The system
objective is therefore to solve the constrained optimization problem
$\min\_{\tau, \mathcal{G}} \mathrm{UCB} \left(\mathcal{L}(\tau,
\mathcal{G})\right)$ subject to $\sum\_{i \in \mathcal{D}\_{n,t}}
\sum\_{r \in \mathcal{P}\_i} c\_{\rho(r)} \le B\_t$ and $\sum\_{i \in
\mathcal{D}\_{n,t}} \mathbf{1}\{\rho(r)=\kappa,\; r \in \mathcal{P}\_i\}
\le \Omega\_\kappa,  \forall \kappa$.

---

<sup>1</sup> E.g. deep neural-networks, large language models, novice
and expert human annotators, etc.  
<sup>2</sup> Such characteristics can evolve over time. However, we
assume they are relatively stable in the short-run.  
<sup>3</sup> In reality noise is often heteroscedastic, arising from
resource capabilities and latent task complexity.  
<sup>4</sup> As reflected by $\mathrm{tr}(\mathbf{N}\_\kappa) \gg
\mathrm{tr}(\mathbf{N}\_{\mathrm{HF}})$.
<sup>5</sup> For human annotators, $v\_{ir}$ is typically derived from
structured ordinal responses $S\_{ir\ell}$ over a template $\mathcal{Q}$.  
<sup>6</sup> Task difficulty is frequently proxied by agreement
($\alpha\_i$) calculated over an initial rater subset.  
<sup>7</sup> E.g. prediction-power-inference (PPI), double-sampling, etc.  
<sup>8</sup> $a\_\tau(x)=1$ corresponds to no demotion and $a\_\tau(x)=0$
corresponds to complete filtering.  
<sup>9</sup> Where explainability is required, $g(v\_{ir})$ may be
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
$x, x\_i$ — Entity in the population  
$i$ — Entity index  
$t$ — Time index  
$Y\_i$ — Latent true label (1 = positive class)  
$E\_{it}$ — Engagement weight of entity $x\_i$ at time $t$  
$p\_t(x)$ — Nominal (traffic-weighted) distribution, $p\_t(x) \propto E\_{it}$  
$\mu\_E = \mathbb{E}\_{x \sim p\_t}[Y(x)]$ — Engagement-weighted prevalence  

### Sampling

$X\_i$ — Sampled entity  
$R\_{it}$ — Sampling indicator  
$\pi\_{it} = \mathbb{P}(R\_{it}=1)$ — Inclusion probability  
$\mathcal{D}\_{n,t}$ — Sampled dataset $\{X\_i \mid R\_{it}=1\}$  
$q(x)$ — Importance sampling distribution  
$w\_i = \frac{p\_t(X\_i)}{q(X\_i)}$ — Importance weight  
$\mathcal{H}$ — Set of strata  
$h$ — Stratum index  
$T$ — Time horizon for pooled sampling  

### Resources and Policies

$\mathcal{R}$ — Set of resources  
$r$ — Resource index  
$\mathcal{K}$ — Set of resource types  
$\kappa$ — Resource type index  
$\rho(r)$ — Mapping from resource to type  
$\Phi\_{\kappa} = \langle c\_\kappa, \Omega\_\kappa, N\_{\kappa} \rangle$ — Resource profile 
$c\_{\kappa}$ — Marginal cost  
$\Omega\_\kappa$ — Processing capacity  
$\mathbf{N}\_\kappa$ — Noise structure  
$B\_t$ — Total budget  
$\mathcal{G}$ — Labeling policy  

### Annotation and Valuation

$\mathcal{P}\_i \subset \mathcal{R}$ — Panel of resources  
$v\_{ir}$ — Valuation  
$\mathcal{A}(\cdot)$ — Aggregation function  
$\hat{V}\_i = \mathcal{A}(\{v\_{ir}\})$ — Aggregated valuation  
$\eta$ — Decision threshold  
$\hat{Y}\_i = \mathbb{I}(\hat{V}\_i > \eta)$ — Predicted label  
$\tilde{Y}\_i$ — Noisy reference label  
$\mathcal{D}\_{n^{(2)}}$ — High-fidelity subsample  

### Structured Human Annotation

$\mathcal{R}^{(H)}$ — Human annotators  
$\mathcal{Q}$ — Annotation questions  
$\ell$ — Question index  
$S\_{ir\ell}$ — Ordinal response  
$v\_{ir}$ — Derived valuation  

### Prediction and Deployment

$v(x)$ — Scalable valuation  
$g(\cdot)$ — Calibration  
$a\_\tau(x) \in [0,1]$ — Exposure multiplier  
$\tau$ — Deployment policy  
$p\_{t+1} = \mathcal{F}(p\_t, \tau\_t)$ — Population dynamics  

### Evaluation

$\mathbf{M}(\eta)$ — Weighted confusion matrix  
$p\_{\omega}(\mathbf{M})$ — Weighted precision  
$r\_{\omega}(\mathbf{M})$ — Weighted recall  
$\hat{E}\_{it}$ — Counterfactual engagement  
$\mathcal{M}$ — Protected subset  
$\Delta\_{\mathcal{M},\mathcal{X}}\mathrm{FPR}$ — FPR difference  

### Objective and Optimization

$\mathcal{L}(\tau,\mathcal{G})$ — Loss functional  
$\mathrm{UCB}(\cdot)$ — Upper confidence bound  
$\min\_{\tau,\mathcal{G}} \mathrm{UCB}(\mathcal{L})$ — System objective  
$n\_\kappa$ — Assignments per type  

### Uncertainty

$\hat{\theta}$ — Estimator  
$\mathcal{B}$ — Bootstrap operator  
$b$ — Bootstrap index  
$\mathcal{D}^{(b)}\_{n,t}$ — Resampled dataset  
$\hat{\theta}^{(b)}$ — Bootstrap estimate  

### Causal Variables

$W$ — Treatment  
$Z\_i$ — Covariates
