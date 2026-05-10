# HarnessGen: Diagnosis-Triggered Structural Adaptation for LLM Agent Harnesses

LLM agent capability increasingly depends on the harness surrounding the model:
the runtime structure that manages context, state, tools, verification,
recovery, and control flow. Recent work on harness self-improvement optimizes
prompts, modules, workflows, or harness code, but often treats recurring
failures as signals for further tuning rather than as evidence that a new
runtime computation stage may be useful. We present **HarnessGen**, a
framework for diagnosis-triggered structural adaptation of LLM agent harnesses.
HarnessGen clusters recurring development traces, tests whether existing
harness components can adapt through bounded recalibration, and grows new typed
runtime modules only when persistent failure clusters remain unresolved under
that budget. Candidate modules are integrated through schema-constrained
graph-surgery primitives, validated in sandboxed replay, and managed by an
atrophy-and-pruning lifecycle that limits structural bloat. Experiments on
Terminal-Bench 2.0, SWE-bench Verified, SWE-bench Multilingual,
OSWorld-Verified, and $\tau^2$-Bench show that HarnessGen improves over
recalibration-only and ungated structural-search baselines on the primary
in-domain benchmarks, outperforms matched-budget harness-search baselines on
TB2, SWE-bench, and $\tau^2$-Bench, and remains competitive on
OSWorld-Verified while maintaining low regression on previously solved tasks.
These results support diagnosis-triggered structural adaptation as a practical
mechanism for building more adaptive and reliable LLM agent harnesses.

## Contributions

- We formulate harness adaptation as an operational distinction between
  *recalibration-addressable* and *structure-triggering* failure
  regimes, identified through recurring trace clustering and a bounded
  recalibration gate.

- We introduce **HarnessGen**, a framework that inserts typed modules
  through schema-constrained proposal, safe graph surgery, sandboxed
  replay validation, and atrophy-based pruning.

- We evaluate HarnessGen on Terminal-Bench 2.0,
  SWE-bench Verified, SWE-bench Multilingual, OSWorld-Verified, and
  $\tau^2$-Bench Telecom, showing that diagnosis-triggered structural
  adaptation improves over recalibration-only and ungated graph-search
  baselines on the primary in-domain benchmarks, outperforms harness-search
  baselines on TB2, SWE-bench, and $\tau^2$-Bench, and remains competitive on
  OSWorld-Verified.

## HarnessGen

![HarnessGen Framework](resource/HarnessGen.pdf)

## Methodology

### Harness Representation

We define an agent harness $\mathcal{H}$ as a directed acyclic computation
graph over typed modules representing the *intra-step* computation at
each agent step; the outer observe--act loop remains a separate recurrent
structure.

Formally, $\mathcal{H} = (\mathcal{O}, \mathcal{E}, \Theta, \Phi)$,
where $\mathcal{O}$ is the module set,
$\mathcal{E} \subseteq \mathcal{O} \times \mathcal{O}$ the directed edge set,
$\Theta$ module-level parameters, and $\Phi$ activation functions.
Each module $o_i$ is a typed tuple
(`name`, `input_schema`, `output_schema`,
`executor`, `side_effects`, `budget`), where
`executor` may be an LLM prompt, deterministic function, tool wrapper,
or composite routine.
Each activation function $\phi_i: \mathcal{S} \rightarrow [0,1]$ is
thresholded deterministically at evaluation time as
$$
b_i^t = \mathbf{1}[\phi_i(s_t) \geq \theta_i^{\text{act}}]
$$
unifying mandatory and conditionally activated modules within the same
framework.

At each step $t$, the harness resolves the active subgraph
$\mathcal{G}_t = \{o_i \in \mathcal{O} : b_i^t = 1\}$, executes modules in
topological order, and passes intermediate states along edges.
The execution trajectory on task $\tau$ is
$\xi = (s_0, \mathcal{G}_0, a_0, \ldots, s_T)$, with binary outcome
$r(\xi, \tau) \in \{0,1\}$.

The empirical performance of body plan
$\mathcal{B} = (\mathcal{O}, \mathcal{E})$ on cluster $\mathcal{T}_c$ is:

$$
    R(\mathcal{B}, \Theta, \mathcal{T}_c) = \frac{1}{|\mathcal{T}_c|}
    \sum_{\tau \in \mathcal{T}_c}
    \mathbb{E}_{\xi \sim \mathcal{H}(\tau)}\left[ r(\xi, \tau) \right].
$$

A body plan is *empirically under-expressive* on $\mathcal{T}_c$ if,
after bounded recalibration $\mathcal{R}(\mathcal{B}, \mathcal{T}_c)$
exhausting a fixed budget, performance remains below threshold $\epsilon_\text{struct}$:

$$
    R\!\left(\mathcal{B},\, \hat{\Theta}_{\mathcal{R}},\, \mathcal{T}_c\right)
    < \epsilon_\text{struct},
    \quad \hat{\Theta}_{\mathcal{R}} = \mathcal{R}(\mathcal{B}, \mathcal{T}_c).
$$

This definition is deliberately operational: it replaces an unverifiable
maximization with a practical recalibration budget, making the distinction
between miscalibration and structural-gap failures experimentally decidable
within a specified compute envelope.

### Development Cycle Overview

HarnessGen alternates between an *execution loop* that runs tasks and
collects trajectory data, and a *development loop* that analyzes failures
and updates the body plan.

### Failure Diagnosis

For each failure trajectory $\xi^{(m)}$ we compute a structured descriptor
$d^{(m)} = \psi(\xi^{(m)}) \in \mathbb{R}^D$ decomposed as:

$$
    d^{(m)} = \left[ d_\text{topo}^{(m)},\; d_\text{timing}^{(m)},\;
    d_\text{state}^{(m)},\; d_\text{tool}^{(m)},\;
    d_\text{outcome}^{(m)} \right],
$$

encoding the module activation sequence, per-module latency and divergence
step, context length and verifier disagreement, tool invocation patterns and
error codes, and terminal failure type.

Clusters at cycle $k$ are matched to those at cycle $k-1$ via the Hungarian
algorithm on symmetric KL divergences between fitted Gaussian components.
A cluster is *persistent* if a matched descendant has been non-empty for
$\kappa$ consecutive cycles.

For each cluster we execute the *bounded recalibration test*, a
structured search over parameter space of existing modules---including
trajectory-reflective prompt optimization, trigger threshold adjustment,
routing weight update, and few-shot demonstration selection---within a fixed
budget. The procedure is prohibited from inserting new modules or altering
graph topology.

A cluster is a structural-gap candidate when the residual failure gap
$$
\Delta_c = \epsilon_\text{struct} - R(\mathcal{B}_k, \hat{\Theta}_\mathcal{R}^{(c)}, \mathcal{T}_c)
$$
is positive *and* the cluster is persistent:

$$
    \text{SG}(c, k) = \mathbf{1}\!\left[\Delta_c > 0\right] \cdot
    \mathbf{1}\!\left[\text{Persistent}(c, k)\right].
$$

### Module Proposal and Control-Flow Integration

For each cluster satisfying $\text{SG}(c,k) = 1$, the developmental
controller generates a module proposal instantiating a module from a typed
schema library of validated archetypes---verifiers,
state compressors, triage modules, constraint/ambiguity resolvers, and
recovery handlers---rather than synthesizing arbitrary code.

The primary module families are:

- Verifier
- Constraint / ambiguity resolver
- State compressor / memory distiller
- Tool triage
- Recovery handler

Each proposal is a five-tuple
$\rho_c = (\hat{o}_c,\; \hat{\phi}_c,\; v_c,\; \mathcal{S}_c,\; \Lambda_c)$,
where $\hat{o}_c$ is the candidate module,
$\hat{\phi}_c$ the proposed activation function,
$v_c$ the insertion vertex,
$\mathcal{S}_c$ the applicable task scope,
and $\Lambda_c$ the lifecycle policy.

Harness integration supports four graph-surgery primitives:

- `pre-insert`
- `post-insert`
- `guard-insert`
- `branch-insert`

All primitives preserve DAG acyclicity and enforce
schema type-compatibility at both insertion edges.

The empirical utility of a proposal is:

$$
    \hat{U}(\rho_c) = \frac{1}{|\mathcal{B}_\text{rep}|}
    \sum_{\tau \in \mathcal{B}_\text{rep}} \left[
    r(\xi_{\mathcal{H}_{+\hat{o}_c}}, \tau)
    - r(\xi_{\mathcal{H}_k}, \tau) \right]
    - \lambda_\text{cost} \cdot \overline{\Delta t}(\hat{o}_c),
$$

penalizing latency overhead. A proposal is accepted when
$\hat{U}(\rho_c) > \eta$ and the regression rate
$\hat{R}_\text{reg}(\rho_c) \leq \epsilon_\text{reg}$
on a held-out set of previously solved tasks.

### Lifecycle Management

Constitutional modules $\{o_\text{obs}, o_\text{plan}, o_\text{act},
o_\text{resp}\}$ are excluded from atrophy; lifecycle management applies only
to grown modules. Each grown module maintains a utility signal:

$$
    u_i^{(k)} = (1-\beta)\, u_i^{(k-1)} + \beta \cdot \hat{\alpha}_i^{(k)},
    \quad
    \hat{\alpha}_i^{(k)} = \frac{1}{|\tilde{\mathcal{F}}_k^i|}
    \sum_{\tau \in \tilde{\mathcal{F}}_k^i}
    \left[ r(\xi_\mathcal{H}, \tau) - r(\xi_{\mathcal{H}_{-i}}, \tau) \right],
$$

where counterfactual reruns use deterministic decoding and a fixed seed.
When the signal falls below $u_\text{min}$ for $W$ consecutive cycles,
the module enters a three-stage lifecycle:

1. *activation decay*
2. *shadow mode*
3. *pruning*

Shadow mode confirms that pruning does not degrade performance before removal.
