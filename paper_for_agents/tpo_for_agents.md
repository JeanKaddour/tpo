# Target Policy Optimization

**Author:** Jean Kaddour

**Code:** https://github.com/JeanKaddour/tpo

---

## Abstract

In RL, given a prompt, we sample a group of completions from a model and score them. Two questions follow: which completions should gain probability mass, and how should the parameters move to realize that change? Standard policy-gradient methods answer both at once, so the update can overshoot or undershoot depending on the learning rate, clipping, and other optimizer choices. We introduce *Target Policy Optimization* (TPO), which separates the two questions. Given scored completions, TPO constructs a target distribution $q_i \propto p_i^{\,\mathrm{old}} \exp(u_i)$ and fits the policy to it by cross-entropy. The loss gradient on sampled-completion logits is $p^\theta - q$, which vanishes once the policy matches the target. On tabular bandits, transformer sequence tasks, and billion-parameter LLM RLVR, TPO matches PG, PPO, GRPO, and DG on easy tasks and substantially outperforms them under sparse reward.

![TPO matches baselines on easy tasks and outperforms them under sparse reward.](figures/hero.png)
*Figure 1: TPO matches baselines on easy tasks and outperforms them under sparse reward. (a) On an MNIST contextual bandit with dense reward, TPO converges slightly faster than GRPO and DG. (b) On a sparse-reward token-reversal task (reward only at end of sequence), GRPO and DG stall near random while TPO solves the task. Both panels show mean ± s.e. over 20 seeds.*

---

## 1. Introduction

Consider a prompt for which we sample a small group of candidate completions from a model and score them. We want to shift probability mass toward the better completions. Standard policy-gradient methods entangle the desired redistribution with the optimizer mechanics that realize it. This coupling can make learning fragile, especially when reward is sparse (Figure 1).

A natural fix is to decouple the two questions: first construct a target distribution that encodes the desired redistribution, then fit the policy to it. This reweight-then-fit idea dates to [dayan1997using] and has been instantiated by REPS [peters2010reps] and MPO [abdolmaleki2018mpo], but those methods require learned Q-functions and constrained optimization over action spaces.

We propose *Target Policy Optimization* (TPO), which applies the same principle to the finite candidate sets used in group-based RL. In this setting, the target distribution is available in closed form, without a critic or dual optimization. Given the probabilities $p_i^{\mathrm{old}}$ assigned by the behavior policy and standardized scores $u_i$, TPO constructs $q_i \propto p_i^{\mathrm{old}} \exp(u_i)$, then fits the policy to $q$ by cross-entropy. The gradient vanishes exactly when the policy matches the target.

We evaluate TPO on exact tabular bandits, MNIST contextual bandits, sparse-reward transformer tasks, and LLM RLVR. It matches policy-gradient baselines on easier tasks and outperforms them where reward is sparse.

---

## 2. Target Policy Optimization

Let $x$ denote a context (e.g. a state or prompt). For each context, we sample $K$ candidates $y_1,\dots,y_K \sim \pi_{\text{old}}(\cdot \mid x)$ and score them with a scalar scorer $S$. In our on-policy experiments, $\pi_{\text{old}}$ is simply the rollout-time snapshot of the current policy. We standardize the raw scores $s_i = S(x, y_i)$ within each group to obtain $u_i = [\operatorname{standardize}(s)]_i$, mapping the zero-variance case to $u = 0$ (Appendix A).

Let $\ell_i^\theta = \log \pi_\theta(y_i \mid x)$ denote the log-probability the current policy assigns to candidate $i$. The policy over the group is

$$
p_i^\theta = \frac{\exp(\ell_i^\theta)}{\sum_{j=1}^K \exp(\ell_j^\theta)}. \quad \text{(Eq. 1)}
$$

Writing $p_i^{\text{old}}$ for the same quantity under $\pi_{\text{old}}$, frozen at rollout time, we tilt this distribution toward higher-scoring candidates to form the target

$$
q_i = \frac{p_i^{\text{old}} \exp(u_i / \eta)}{\sum_{j=1}^K p_j^{\text{old}} \exp(u_j / \eta)}, \quad \text{(Eq. 2)}
$$

where $\eta > 0$ is a temperature (we use $\eta = 1$ throughout; Appendix D shows this is robust).

We fit the policy to this target by minimizing the cross-entropy

$$
\mathcal{L}_{\text{TPO}}(\theta) = -\sum_{i=1}^K q_i \log p_i^\theta, \quad \text{(Eq. 3)}
$$

treating $q$ as fixed. The loss gradient satisfies $\partial \mathcal{L}/\partial \ell_i^\theta = p_i^\theta - q_i$, so gradient descent moves in direction $q_i - p_i^\theta$ and vanishes once the policy matches the target.

In the on-policy setting, the full update takes a few lines of code (see code snippets below). If rollouts are reused for additional optimization epochs, $q$ stays frozen while the $\log p$ term is recomputed under $\theta$.

**Why standardize.** The target (Eq. 2) exponentiates the scores, so groups with the same ranking but different numerical spread would produce very different targets. For example, $(1, 0, -1)$ and $(100, 0, -100)$ express the same ordering, but exponentiating $(100, 0, -100)$ makes the target nearly deterministic while $(1, 0, -1)$ yields a gentle tilt. Standardization makes the update depend on relative within-group performance rather than arbitrary score units, and largely removes the need to tune $\eta$.

**KL-regularized interpretation.** The target $q$ is equivalently the unique solution of

$$
q = \arg\max_{r \in \Delta^{K-1}} \left\{ \sum_{i=1}^K r_i u_i - \eta\,\mathrm{KL}(r \,\|\, p^{\text{old}}) \right\}, \quad \text{(Eq. 4)}
$$

where $\Delta^{K-1}$ is the simplex over the sampled candidates.

**Proposition 1.** *Assume $p_i^{\text{old}} > 0$ for every sampled candidate. Then the target in Eq. 2 is the unique maximizer of Eq. 4. Furthermore, treating $q$ as fixed, the cross-entropy loss in Eq. 3 satisfies $\nabla_{\ell^\theta}\mathcal{L}_{\text{TPO}} = p^\theta - q$, so the unique stationary distribution over the sampled candidates is $p^\theta = q$.*

**Proof.** The objective in Eq. 4 is strictly concave in $r$ because $-\mathrm{KL}(r \,\|\, p^{\text{old}})$ is strictly concave on the simplex when $p^{\text{old}}$ has full support. Introducing a Lagrange multiplier for $\sum_i r_i = 1$ and differentiating gives

$$
u_i - \eta\Bigl(\log \frac{r_i}{p_i^{\text{old}}} + 1\Bigr) + \lambda = 0,
$$

hence $r_i = C\,p_i^{\text{old}}\exp(u_i/\eta)$ for a normalization constant $C$, which yields Eq. 2.

Treating $q$ as fixed, differentiating the softmax cross-entropy with respect to the group logits gives $\partial \mathcal{L}/\partial \ell_i^\theta = p_i^\theta - q_i$. Therefore $\nabla_{\ell^\theta}\mathcal{L}_{\text{TPO}} = 0$ iff $p^\theta = q$, which identifies the unique stationary distribution over the sampled candidates. ∎

### Algorithm 1: Target Policy Optimization (TPO)

**Input:** Policy $\pi_\theta$, scorer $S$, candidates per context $K$, temperature $\eta$ (default 1).

Repeat until converged:
1. Freeze the behavior policy: $\pi_{\text{old}} \leftarrow \pi_\theta$.
2. Sample a batch of contexts $x$ and candidates $\{y_i\}_{i=1}^K \sim \pi_{\text{old}}(\cdot \mid x)$.
3. Compute scores $s_i = S(x, y_i)$ and form $s=(s_1,\dots,s_K)$.
4. Standardize: $u_i = [\operatorname{standardize}(s)]_i$.
5. Compute the target $q_i = \dfrac{p_i^{\text{old}} \exp(u_i / \eta)}{\sum_{j=1}^K p_j^{\text{old}} \exp(u_j / \eta)}$.
6. Take one or more gradient steps on $\mathcal{L}_{\text{TPO}}(\theta) = -\sum_{i=1}^K q_i \log p_i^\theta$, treating $q$ as fixed.

### Implementation sketch

`log_scores` contains the policy log-probabilities of the sampled candidates, renormalized by `log_softmax` to form the policy over the group; `u` contains standardized task scores; `eta` is an optional temperature with default value 1. The sketch shows the simplest on-policy implementation, where the same `log_scores` tensor is used both to form $q$ and to compute `log_p`, with $q$ detached from the computation graph before the update.

**JAX:**

```python
def tpo_target(log_scores, u, eta=1.0):
    return jax.nn.softmax(
        jax.nn.log_softmax(log_scores, -1)
        + u / eta, -1)

q = jax.lax.stop_gradient(
    tpo_target(log_scores, u))
log_p = jax.nn.log_softmax(log_scores, -1)
loss = -(q * log_p).sum(-1).mean()
```

**PyTorch:**

```python
def tpo_target(log_scores, u, eta=1.0):
    return F.softmax(
        F.log_softmax(log_scores, -1)
        + u / eta, -1)

q = tpo_target(
    log_scores, u).detach()
log_p = F.log_softmax(log_scores, -1)
loss = -(q * log_p).sum(-1).mean()
```

---

## 3. Experiments

Baselines are PPO [schulman2017ppo], GRPO [shao2024deepseekmath], and DG [osband2026dg]. For dense-reward experiments, we compare token-level grouped variants (TPO_token, GRPO_token) that sample $K{=}8$ next-token candidates at each prefix state; for terminal reward, we use sequence-level TPO and GRPO with $K{=}8$ full rollouts per prompt; for LLM RLVR, $K{=}16$. PPO, GRPO, and TPO take multiple gradient epochs per rollout batch; DG uses a single epoch, following [osband2026dg], because it diverges with more (Appendix E). Our GRPO baseline uses the clipped surrogate with $z$-scored advantages and a reverse-KL penalty (Appendix F); we refer to it simply as GRPO throughout.

Where grouped methods consume $K\times$ more rollouts than single-sample methods for the same number of prompts, we report two comparisons. *Prompt-matched*: same number of prompts; grouped methods use more total rollouts. *Interaction-matched*: same total rollouts; single-sample methods see more prompts.

Unless stated otherwise, all transformer experiments use Optax's [deepmind2020jax] Muon optimizer [jordan2024muon] at learning rate $10^{-3}$ and batch size $B{=}100$, with Muon applied to 2D parameter tensors and AdamW to non-2D tensors.

### 3.1 Single-context bandit: within-context update quality

Following [osband2026dg], we replace the network with explicit logit tables so the softmax policy and its gradients can be computed exactly. These tabular runs do not use a neural optimizer; we take normalized logit steps of size $\alpha{=}0.1$ directly.

We consider a $K$-armed bandit with one correct action $y^*$ among $K{=}100$ choices. The reward is $R = \mathbf{1}\{A = y^*\}$. At each step, the agent samples $B{=}100$ actions, computes a gradient estimate, and takes a normalized step. We average over 100 seeds.

![Single-context symmetric bandit.](figures/tabular_single.png)
*Figure 2: Single-context symmetric bandit ($K{=}100$, $B{=}100$, normalized steps). (a) TPO and DG converge fastest; GRPO and PG plateau at higher error. (b) TPO maintains the lowest misalignment to the oracle gradient throughout training.*

Figure 2 shows that TPO and DG converge fastest. Unlike PG and GRPO, they continue improving beyond 1% error. The misalignment panel shows why: TPO stays closest to the oracle policy-gradient direction as the policy concentrates, while GRPO becomes increasingly misaligned.

### 3.2 Multi-context bandit: cross-context allocation

The single-context experiment tests update quality; this one tests how a normalized update allocates a finite step budget *across* contexts. We consider $N{=}100$ independent contexts, each a $K{=}10$ bandit with $\mathcal{N}(0,1)$ logit initialization [osband2026dg]. Exact population updates remove sampling variance, so any remaining gap reflects how each method distributes the step. We include the cross-entropy (CE) oracle, which is optimal under normalized steps in this setting.

![Multi-context bandit.](figures/tabular_multi.png)
*Figure 3: Multi-context bandit ($N{=}100$, $K{=}10$, exact gradients). (a) All methods converge; the CE oracle is fastest. (b) TPO achieves near-zero misalignment to the CE oracle direction, confirming its update direction targets the optimal allocation.*

Figure 3 shows that all methods eventually converge and that CE is fastest, but among the RL updates TPO is the closest to CE in both error and direction. DG and GRPO improve slightly faster at the start, but TPO overtakes them after the early transient and finishes with the lowest error of the three. The misalignment panel shows the same pattern more clearly: TPO remains much closer to the CE direction throughout training.

This pattern is analytically transparent in the one-hot setting. Let $p_n=\pi_n(y_n)$ be the current probability of the correct action in context $n$. Working in *logit space* with baseline $b{=}0$, every exact update can be written as

$$
g_n = \beta(p_n)\,(e_{y_n}-\pi_n),
$$

so all methods share the same within-context direction $e_{y_n}-\pi_n$ and differ only in the scalar weight $\beta(p_n)$. Because the global step is normalized, $\beta$ controls how much of that step is spent on context $n$: a method that assigns larger $\beta$ to easy (high-$p_n$) contexts wastes budget where it is least needed.

The coefficients (derived in Appendix B) are:

$$
\beta_{\mathrm{CE}}(p_n)=1,\qquad
\beta_{\mathrm{DG}}(p_n)=\frac{p_n}{1+p_n},\qquad
\beta_{\mathrm{GRPO}}(p_n)=\sqrt{\frac{p_n}{1-p_n}},\qquad
\beta_{\mathrm{TPO}}(p_n)=\frac{p_n(\lambda-1)}{1-p_n+\lambda p_n},
$$

where $\lambda=\exp(u_{y_n}-u_{a\neq y_n})\approx 28$ for $K{=}10$.

CE treats every context equally ($\beta{=}1$ everywhere). DG and GRPO both *vanish* as $p_n \to 0$: when a context is hard, they barely update it. DG vanishes linearly ($\beta \approx p_n$) and GRPO vanishes as $\sqrt{p_n}$, so both spend most of the normalized step on contexts that are already nearly solved. TPO's coefficient, by contrast, stays large even at small $p_n$: at $p_n{=}0.1$, $\beta_{\mathrm{TPO}}{=}0.73$ versus $0.09$ for DG and $0.33$ for GRPO. TPO therefore allocates more update budget to hard contexts, which is why it tracks the CE oracle more closely and overtakes the scalar-weighted baselines after the initial transient.

### 3.3 Neural policy learning: MNIST contextual bandit

Following [osband2026dg], we cast MNIST classification as a one-step contextual bandit: the agent samples $A \in \{0,\dots,9\}$ and receives $R = \mathbf{1}\{A = Y\}$ without observing the label $Y$. A two-layer ReLU network trains for 10,000 steps (20 seeds). Each method samples a single action per context and updates from bandit feedback alone. We optimize the network with Adam at learning rate $10^{-3}$ and batch size $B{=}100$.

![MNIST contextual bandit: TPO converges fastest and reaches the lowest error.](figures/mnist.png)
*Figure 4: MNIST contextual bandit: TPO converges fastest and reaches the lowest error. (a) Learning curves for all single-sample bandit updates, including the same-signal ablation Group PG. (b) At step 2,000, for each misclassified example we measure how much more each method increases the true-class probability $p_y$ compared to a generic one-vs-rest baseline (Appendix C), binned by wrong-class concentration $c = \max_{j \ne y}\pi_j/(1-\pi_y)$. TPO's extra gain grows with concentration; DG's does not.*

Figure 4 shows that the tabular pattern survives the transition to a neural policy: TPO converges fastest (5% error at step 1,600 vs. 2,200 for DG) and reaches the lowest final error (2.9%). With a single sampled action per context, GRPO reduces to batch-normalized REINFORCE and therefore performs comparably to PG (5.9% vs. 5.3%).

PG, single-sample GRPO, and Group PG all learn "increase the true class versus the rest" without using which wrong class was sampled — in expectation, they collapse to a rescaled one-vs-rest direction $c(x)(e_y-\pi)$ (Appendix C). DG and TPO both condition on the sampled action, but only TPO turns a failed sample into a class-specific target update: a correct sample pulls probability toward the label, while an incorrect sample directly suppresses the sampled wrong class. This extra structure should matter most when error mass is concentrated on one or a few confusing alternatives. Removing it confirms this: Group PG keeps the same candidates and standardized scores but replaces target matching with scalar-weighted REINFORCE, raising final error from 2.9% to 7.2%.

Figure 4(b) tests that prediction directly. On each misclassified test example, let $c = \max_{j \ne y}\pi_j/(1-\pi_y)$ denote the fraction of wrong-class mass carried by the most likely wrong label.

We then compare the exact first-order gain in $p_y$ to the scalar one-vs-rest surrogate from Appendix C. TPO's surplus is near zero when the error mass is diffuse, but rises to $0.073$ in the highest-concentration bin at the step-2,000 checkpoint; DG stays slightly negative throughout. TPO's benefit therefore appears exactly where one-vs-rest corrections are too coarse: examples dominated by one confusing wrong label.

### 3.4 Dense sequence reward: token-level transformer grouping

Dense per-token rewards let us group at the token level. We use the Token Reversal task of [osband2026dg]: a 2-layer, 4-head causal transformer autoregressively reverses an input sequence of length $H{=}10$ drawn uniformly from a vocabulary of size $V$. The reward is the *bag-of-tokens* fraction of tokens reversed correctly. We sweep $V \in \{2, 4, 8, 16\}$, growing the output space from $2^{10} \approx 10^3$ to $16^{10} \approx 10^{12}$, and report *sequence error* (fraction of tokens incorrect) averaged over 20 seeds.

At each prefix state, we sample $K{=}8$ next-token candidates and form the group over those candidates (TPO_token, GRPO_token). For autoregressive models, $\ell_i^\theta$ is the usual sum of per-token log-probabilities. All methods follow one behavior trajectory per prompt, so environment interactions are matched.

![Token Reversal.](figures/transformer_vocab_sweep.png)
*Figure 5: Token Reversal (bag-of-tokens reward, $K{=}8$ token candidates). All methods use $B{=}100$ prompts and follow one behavior trajectory each; TPO_token and GRPO_token additionally sample $K$ next-token candidates at each prefix state. Columns vary vocabulary size $V \in \{2, 4, 8, 16\}$.*

The gap between methods widens with task difficulty (Table 1, Figure 5): at $V{=}16$, TPO_token reaches 1% error at step 102, compared to 148 for GRPO_token, 259 for PPO, and 393 for DG.

**Table 1: Steps to 1% error.** Token Reversal (bag-of-tokens reward, $K{=}8$ token candidates). **Bold**: fastest method at each $V$. All methods use the same environment interactions per step.

| Method       | $V=2$    | $V=4$    | $V=8$    | $V=16$   |
|--------------|----------|----------|----------|----------|
| TPO_token    | **58**   | **74**   | **103**  | **102**  |
| GRPO_token   | 904      | 141      | 124      | 148      |
| DG           | 199      | 273      | 314      | 393      |
| PPO          | 872      | 181      | 191      | 259      |

Because all methods follow a single behavior trajectory per prompt, there is no prompt-matched vs. interaction-matched distinction, rollout budgets are identical. GRPO_token improves with larger $V$ (where more token candidates provide a richer signal) but lags behind TPO_token throughout. DG and PPO, which lack within-group structure, scale less favorably.

### 3.5 Generalization across task and reward variants

Does the pattern hold beyond token reversal? Following [osband2026dg], we evaluate four target logics (copy, flip, reverse copy, reverse flip) under two reward structures (bag-of-tokens and sequential), yielding eight variants. Sequential reward gives credit only up to the first incorrect token, sparser than bag-of-tokens but denser than terminal. Hyperparameters match Section 3.4 ($H{=}10$, $V{=}2$, $K{=}8$ token candidates); 10 seeds, 1,000 episodes.

![Task variations, prompt- and interaction-matched.](figures/transformer_variations_combined.png)
*Figure 6: Task variations, prompt- and interaction-matched. Top two rows: prompt-matched. Bottom two rows: interaction-matched. Within each pair, the first row is bag-of-tokens reward and the second is sequential reward. Columns vary target logic.*

Under bag-of-tokens reward (top row of Figure 6), TPO_token reaches 1% error first on all eight variants (Table 2), 2–6× faster than the runner-up. All methods except PPO eventually reach 1% on bag-of-tokens tasks. Under sequential reward, TPO_token's advantage widens: it reaches 1% error on all four tasks within our budget; DG converges on all four but more slowly; GRPO_token and PPO fail to converge on any.

**Table 2: Steps to 1% error, task variations** ($K{=}8$ token candidates). **Bold**: fastest per row. "−": never reached within budget.

| Reward        | Target        | TPO_token | GRPO_token | DG  | PPO |
|---------------|---------------|-----------|------------|-----|-----|
| Bag of tokens | Copy          | **81**    | 338        | 219 | 170 |
| Bag of tokens | Flip          | **56**    | 104        | 201 | 146 |
| Bag of tokens | Rev. copy     | **55**    | 352        | 202 | −   |
| Bag of tokens | Rev. flip     | **59**    | 209        | 200 | 143 |
| Sequential    | Copy          | **295**   | −          | 439 | −   |
| Sequential    | Flip          | **321**   | −          | 349 | −   |
| Sequential    | Rev. copy     | **159**   | −          | 515 | −   |
| Sequential    | Rev. flip     | **276**   | −          | 309 | −   |

Under sequential reward, only TPO_token and DG converge. The key is per-state targeting: under sequential reward, prefixes after the first mistake see zero reward for every candidate, so the target there matches the old policy and introduces no spurious signal. TPO_token therefore concentrates its update on informative prefixes where at least one candidate continues correctly. DG's sigmoid gating also helps but is slower; GRPO_token and PPO lack an equally explicit local target.

### 3.6 Sparse credit assignment: terminal reward

The hardest credit-assignment test removes intermediate feedback entirely: the model receives an exact-match reward only after the full sequence. Without per-token rewards, we revert to sequence-level TPO and GRPO, each sampling $K{=}8$ complete rollouts per prompt. Prompt-matched runs use $B{=}100$; interaction-matched runs scale single-sample batch size and learning rate by $K$ and $\sqrt{K}$ respectively. Other hyperparameters match Section 3.4 ($V{=}2$); we sweep $H \in \{7, 8, 9, 10\}$ over 2,000 episodes. We report exact-match error (fraction of sequences with any mistake), not token-level error.

![Terminal reward, prompt- and interaction-matched.](figures/rlvr_sweep_combined.png)
*Figure 7: Terminal reward, prompt- and interaction-matched. Top row: prompt-matched ($B{=}100$ for all methods). Bottom row: interaction-matched ($B{\cdot}K{=}800$ rollouts per step, with single-sample batch size and learning rate scaled by $K$ and $\sqrt{K}$ respectively). Here grouped methods use $K{=}8$ candidates per prompt. Y-axis: exact-match error. TPO has the lowest error at each $H$ under both matching conditions.*

Under prompt matching, the methods diverge most (Table 3, top row of Figure 7): TPO attains the lowest error at each tested $H$. GRPO and PPO make progress at shorter lengths but degrade steeply; DG fails earlier still. Removing GRPO's KL penalty ($\beta{=}0$) makes it substantially worse (66.6% at $H{=}7$ and no meaningful learning beyond $H{=}8$), showing that the KL term is GRPO's primary stabilizer under sparse reward.

**Table 3: Exact-match error (%), terminal reward.** **Bold**: best method. "−": >95% (no meaningful learning). Left: prompt-matched. Right: interaction-matched. TPO attains the lowest error at each tested $H$.

| Method       | Prompt $H{=}7$ | Prompt $H{=}8$ | Prompt $H{=}9$ | Prompt $H{=}10$ | Interact. $H{=}7$ | Interact. $H{=}8$ | Interact. $H{=}9$ | Interact. $H{=}10$ |
|--------------|----------------|----------------|----------------|-----------------|-------------------|-------------------|-------------------|--------------------|
| TPO          | **6.9**        | **8.6**        | **6.1**        | **7.4**         | **1.8**           | **2.8**           | **5.3**           | **19.0**           |
| GRPO         | 14.5           | 27.6           | 30.0           | 50.4            | 9.6               | 23.2              | 36.2              | 48.7               |
| GRPO (no KL) | 66.6           | 92.5           | −              | −               | 78.1              | 83.8              | −                 | −                  |
| PPO          | 12.0           | 26.3           | 90.6           | −               | 38.6              | 62.1              | 66.2              | −                  |
| DG           | 33.8           | 58.8           | −              | −               | 47.7              | 69.4              | −                 | −                  |

Under interaction matching (bottom row of Figure 7, right half of Table 3), TPO remains ahead at each $H$. The gap is wider here than in the bag-of-tokens experiments, where interaction matching narrowed it substantially. With terminal reward, the bottleneck is not gradient variance but extracting useful signal from sparse outcomes, the regime where target matching matters most.

### 3.7 Anchor and target-matching ablations

To isolate ingredients of TPO's grouped update, we compare TPO against several prompt-matched variants on the same terminal-reward benchmark ($H \in \{7,8,10\}$, $V{=}2$, $K{=}8$, $B{=}100$, 20 seeds). All methods use the same grouped full-sequence rollouts. "TPO-no-anchor" removes the $p^{\text{old}}$ anchor ($q_i \propto \exp(u_i)$). "Group PG" keeps the same candidates and standardized scores but replaces target matching with scalar-weighted policy gradient. "GRPO (no KL)" removes the reverse-KL penalty ($\beta{=}0$).

![Removing the anchor, KL penalty, or target matching each degrades learning.](figures/rlvr_sweep_ablation_prompt_subset.png)
*Figure 8: Removing the anchor, KL penalty, or target matching each degrades learning. Terminal reward, reverse-copy targets, $V{=}2$, $K{=}8$, $B{=}100$, 20 seeds. Shading shows ±1 s.e.*

Full TPO outperforms every ablation at each sequence length (Figure 8), and the gaps widen with $H$: at $H{=}10$, TPO reaches 7.4% while every ablation exceeds 99%. The old-policy anchor is doing real work: removing it is consistently harmful. Target matching itself also matters: keeping the same candidates and standardized scores but reverting to scalar weighting (Group PG) performs worst. Removing GRPO's KL penalty makes it substantially worse, consistent with Section 3.6.

### 3.8 LLM RLVR: transfer to billion-parameter models

GRPO is the de facto standard for billion-parameter LLM RLVR [lambert2025tulu3pushingfrontiers, guo2025deepseekr1]. Does TPO's advantage transfer to this setting?

We compare TPO and GRPO using the `verl` stack [sheng2024verl] on two models (Qwen3-1.7B [yang2025qwen3] and DeepSeek-R1-Distill-Qwen-1.5B [guo2025deepseekr1]) and three tasks: GSM8K [cobbe2021gsm8k], graph coloring, and Knights & Knaves (both from Reasoning Gym [stojanovski2025reasoninggym]). All runs use $K{=}16$ rollouts per prompt; the paired runs differ only in the policy loss (TPO vs. clipped surrogate with $z$-scored advantages). Implementation details (optimizer, LoRA, hardware) are in Appendix G.

![LLM RLVR.](figures/llm_rlvr.png)
*Figure 9: LLM RLVR. Top row: Qwen3-1.7B. Bottom row: DeepSeek-R1-Distill-Qwen-1.5B. All runs use $K{=}16$ rollouts per prompt. Columns: GSM8K (held-out test accuracy, evaluated every 5 steps), Reasoning Gym graph coloring (train mean score), Reasoning Gym Knights & Knaves (train mean score).*

On GSM8K (Figure 9, left column), TPO learns faster early (reaching 50% accuracy ~10 steps before GRPO on Qwen3-1.7B) but both converge to comparable final accuracy (~85–87%), consistent with TPO's advantage being largest during the learning phase.

On the Reasoning Gym tasks (middle and right columns), we plot train mean score. The gap is starker here: on graph coloring, GRPO fails entirely on Qwen3-1.7B (near-zero score for 300 steps) while TPO reaches ~0.96. On R1-Distill-1.5B, both learn but TPO converges higher (~0.96 vs. ~0.81). Knights & Knaves shows the same pattern. These harder tasks expose TPO's advantage more clearly than GSM8K, where both methods eventually saturate.

---

## 4. What explains TPO's gains under sparse reward?

We identify several reinforcing properties: the gradient self-extinguishes once the policy matches the target, signal concentrates on the few informative groups rather than all-fail batches, and the fixed target supports stable multi-epoch reuse. We examine these in a representative sparse-reward regime ($H{=}8$, $V{=}2$, $K{=}32$, $B{=}256$, 2,000 episodes). We compute per-step diagnostics from the original 10-seed runs; the $K$-sweep, epoch sweep, and masking ablations use 30 seeds.

### 4.1 Does TPO's gradient vanish in practice while GRPO's persists?

Because TPO's gradient vanishes at $p^\theta = q$ (Proposition 1), the update should decay as the policy converges. Policy-gradient methods lack this fixed point.

![TPO's gradient self-extinguishes; GRPO's does not.](figures/ablation_gradient.png)
*Figure 10: TPO's gradient self-extinguishes; GRPO's does not ($H{=}8$, $V{=}2$, $K{=}32$). (a) Gradient L2 norms over training. (b) Per-candidate weight proxy on successful (solid) vs. failed (dashed) candidates: mean target mass $q_i$ for TPO, mean $|A_i|$ for GRPO.*

Figure 10(a) tracks the L2 norm of the first-epoch gradient over training. TPO's gradient spikes during the learning phase then decays to near zero once the policy converges (~episode 300). GRPO maintains persistent gradient norms throughout training, even after its error curve plateaus at 12.7%. GRPO's policy keeps moving even after its error plateaus, rather than settling near a fixed point.

Figure 10(b) shows a per-candidate weight proxy on successful candidates (solid) versus failed ones (dashed). Because the proxy differs between methods (target mass $q_i$ for TPO, advantage magnitude $|A_i|$ for GRPO), the panel is an allocation diagnostic, not a gradient decomposition. TPO rapidly removes weight from failed candidates, whereas GRPO continues assigning nonzero advantage magnitude to failures even late in training. The stronger fixed-point claim comes from panel (a): TPO's gradient norm collapses near zero, while GRPO's does not.

### 4.2 How does TPO allocate signal when informative groups are rare?

With $K{=}32$ candidates and a per-sequence success rate of $(1/V)^H \approx 0.4\%$ at initialization, most groups contain no successful completion.

![Most groups carry no signal early in training; TPO eliminates them fastest.](figures/ablation_zerovar.png)
*Figure 11: Most groups carry no signal early in training; TPO eliminates them fastest ($H{=}8$, $V{=}2$, $K{=}32$). (a) Fraction of groups where all $K$ candidates fail. (b) Fraction of prompts with at least one successful candidate.*

Figure 11(a) shows that roughly 90% of groups are all-fail at the start of training. For TPO, these groups are neutral on the rollout snapshot: zero score variance implies standardized scores $u=0$ (Appendix A), so $q = p^{\text{old}}$ and the grouped-loss contribution on the first epoch is exactly zero. Early in training, TPO therefore spends its first-epoch grouped update on the relatively small fraction of groups that actually distinguish better from worse candidates, namely those containing at least one success. (We show all-fail groups rather than total zero-variance groups because late in training zero variance can also arise from all-success groups, which are not the sparse-reward failure mode of interest.)

In the shared-parameter multi-epoch setting, that neutrality need not persist forever. Once informative groups have moved the policy away from the rollout snapshot, revisiting the same all-fail group yields an anchor term back toward $p^{\text{old}}$. This later-epoch pullback can arise for both TPO and our snapshot-KL GRPO baseline, so zero-variance groups are not permanently ignored. The key property is narrower: on the rollout snapshot, when informative groups are scarce, TPO concentrates its grouped signal on the groups that contain actual ranking information.

As training progresses and the policy improves, the all-fail fraction drops: more groups contain at least one successful candidate (Figure 11(b)). This means a larger fraction of each batch contributes nontrivial target structure. TPO drives the all-fail fraction to near zero quickly, whereas GRPO leaves a larger residual and GRPO (no KL) remains substantially worse.

**Group-size ablation.** Varying $K \in \{4,8,16,32,64\}$ changes two things at once (Figure 12). Larger groups are more likely to contain at least one successful completion, and with binary rewards the same within-group $z$-scoring also makes the grouped update much sharper once a success appears.

We therefore interpret this figure as a joint sensitivity sweep over candidate coverage and grouped-signal sharpness. Across 30 seeds, TPO improves from 8.9% error at $K{=}4$ to 5.2% at $K{=}8$, 5.1% at $K{=}16$, 2.6% at $K{=}32$, and 0.36% at $K{=}64$. GRPO is weaker and less monotonic: 19.4%, 19.8%, 9.2%, 4.4%, and 5.6% across the same sweep. Under this combined change, TPO behaves more smoothly across $K$.

![Group-size sensitivity sweep.](figures/ablation_k_sweep.png)
*Figure 12: Group-size sensitivity sweep ($H{=}8$, $V{=}2$, epochs${=}4$). (a) TPO learning curves: steady improvement as $K$ grows, with the strongest performance at $K{=}64$. (b) GRPO learning curves: larger groups help, but performance remains less stable and less monotonic. (c) Final error vs. $K$: TPO improves from 8.9% at $K{=}4$ to 0.36% at $K{=}64$; GRPO improves from 19.4% at $K{=}4$ to 4.4% at $K{=}32$ and then worsens slightly at $K{=}64$ (5.6%). 30 seeds, shading/bars ±1 s.e.*

**Zero-variance masking.** If zero-variance groups were simply dead weight, an obvious intervention would be to mask them explicitly. We test "GRPO (masked)," which zeros the loss for any group where all $K$ candidates receive the same reward (Figure 13).

In the 30-seed aggregate, masking is strongly harmful: final error rises from 6.3% to 29.7%. This suggests that these groups are not just junk to delete. In the multi-epoch setting, once informative groups have moved the shared policy, revisiting zero-variance groups can provide a useful anchor back toward the rollout snapshot. Removing them therefore makes GRPO markedly worse, and still does not approach TPO, which reaches 0.05% in the same setting.

![Zero-variance masking.](figures/ablation_grpo_fix.png)
*Figure 13: Zero-variance masking ($H{=}8$, $V{=}2$, $K{=}32$, epochs${=}4$). (a) Learning curves: GRPO (zv-masked) is substantially worse than both GRPO and TPO. (b) Final error: masking increases GRPO from 6.3% to 29.7%, while TPO reaches 0.05% without any masking. 30 seeds, shading/bars ±1 s.e.*

### 4.3 Does TPO extract more from rare informative batches across epochs?

TPO's fixed target $q$ provides a stable attractor across gradient epochs: the same batch can be reused safely without the trust-region issues that cause DG to diverge with multiple epochs (Appendix E). Under terminal reward, where informative batches are rare, extracting maximum learning from each one is critical.

![Multi-epoch extraction.](figures/ablation_multiepoch.png)
*Figure 14: Multi-epoch extraction ($H{=}8$, $V{=}2$, $K{=}32$). (a) Error curves: TPO with 4 gradient epochs reaches 0.2% error at episode 400 while TPO with 1 epoch is at 1.1%, roughly 5× faster. Both eventually converge to <0.1%. DG, limited to a single epoch, plateaus at 14%. (b) Gradient norms: TPO (4 ep) gradient decays fastest; TPO (1 ep) shows a delayed spike and slower decay; DG's gradient stays low but persistent.*

Figure 14 compares TPO with 4 gradient epochs versus 1. At episode 400, TPO (4 epochs) has reached 0.2% error while TPO (1 epoch) is at 1.1%, roughly 5× faster early convergence. Both eventually reach <0.1%, confirming that multi-epoch extraction primarily accelerates learning rather than enabling it. DG, limited to a single epoch, plateaus at 14%.

**Epoch-count ablation.** We sweep $\{1,2,4,8,16\}$ gradient epochs for both TPO and GRPO (Figure 15). TPO remains stable across the entire range: final error stays below 2.3% everywhere and is already near zero at 1 and 4 epochs (0.02% and 0.05%). GRPO remains strongly non-monotonic: 1 epoch reaches 4.3%, 2 epochs degrades to 37.6%, 4 epochs improves to 6.3%, 8 epochs reaches 3.3%, and 16 epochs reaches 1.1%. GRPO can reach low error at the right epoch count, but is much more sensitive to this choice.

![Epoch-count ablation.](figures/ablation_epoch_sweep.png)
*Figure 15: Epoch-count ablation ($H{=}8$, $V{=}2$, $K{=}32$). (a) TPO learning curves across epoch counts: all converge smoothly and remain low-error throughout. (b) GRPO learning curves: 2 epochs is the worst setting, while 8 and 16 epochs recover strongly. (c) Final error comparison: TPO stays below 2.3% everywhere; GRPO is strongly non-monotonic (37.6% at 2 epochs, 1.1% at 16 epochs). 30 seeds, shading ±1 s.e.*

No single property explains TPO's sparse-reward advantage. The gradient norm collapses as the policy approaches its target, performance degrades smoothly rather than abruptly when $K$ or epoch count varies, and multi-epoch reuse works without careful tuning. These properties reinforce each other and are absent from the baselines.

---

## 5. Related work

**Target-matching and mirror-descent methods.**
TPO's target (Eq. 2) is the closed-form solution to $\arg\min_q \mathrm{KL}(q \,\|\, p^{\text{old}}) - \tfrac{1}{\eta}\,\mathbb{E}_q[u]$ restricted to $K$ candidates. The closest relatives are REPS [peters2010reps], MPO [abdolmaleki2018mpo], and V-MPO [song2020vmpo], which use the same exponential-tilting step but require a critic or value estimate to supply the improvement signal. AWR [peng2019awr] also uses KL-regularized improvement weights but treats each sample's $\exp(A/\beta)$ as a fixed scalar on its log-likelihood, so its gradient does not self-extinguish at the target. TPO's distinguishing property is that the finite scored candidate set provides the target in closed form without a critic or inner optimization loop, and its gradient $p^\theta{-}q$ vanishes once the target is matched. MDPO [tomar2020mdpo] gives a mirror-descent perspective on the same family. More generally, TPO can be read as a KL-regularized policy-improvement operator on the sampled candidate simplex rather than the full action space [kakade2001natural, geist2019regularized].

**Group-based policy-gradient methods.**
RLOO [ahmadian2024rloo] and GRPO [shao2024deepseekmath] also score multiple candidates per context but convert them into per-sample scalar weights inside a policy-gradient objective. TPO instead builds a target distribution on the candidate simplex and fits the policy to it. Recent GRPO variants address specific failure modes while remaining scalar-weighted PG methods: Dr. GRPO [liu2025drgrpo] removes a difficulty bias from within-group $\sigma$-normalization [murphy2025reinforcementlearningoverview]; DAPO [yu2025dapo] uses asymmetric clipping to prevent entropy collapse; GSPO [zhang2025gspo] fixes a per-token importance-ratio mismatch when rewards are trajectory-level. TPO replaces the scalar-weighted update with a single target distribution over the group, avoiding importance ratios and clipping entirely. Because it still standardizes within-group scores, however, low-variance difficulty-bias effects can remain in principle, as discussed in Section 6.

**Single-sample policy-gradient methods.**
REINFORCE [williams1992simple], TRPO [schulman2015trpo], PPO [schulman2017ppo], REINFORCE++ [hu2025reinforceplusplus], and ReMax [li2024remax] all assign scalar advantage weights to sampled actions. ReMax removes the value model and uses a greedy-decode baseline for variance reduction, yielding large memory and speed gains over PPO, but the gradient remains the standard score-function estimator. DG [osband2026dg] corrects gradient misallocation *across contexts* via sigmoid gating; TPO addresses misallocation *within* a context's candidate set. The two are complementary and can be composed.

**Regression- and preference-based methods.**
REBEL [gao2024rebel] reduces RL to iterative least-squares regression on reward *differences* between paired completions, generalizing NPG with strong agnostic regret bounds. Both REBEL and TPO construct a target from rewards and the behavior policy, but differ in loss and structure: REBEL uses squared loss on log-probability ratios over pairs; TPO uses cross-entropy on a distribution over a candidate group. PMPO [abdolmaleki2025pmpo] is the closest target-matching method: it partitions candidates into accepted/rejected sets and regularizes toward a frozen $\pi_{\text{ref}}$, whereas TPO keeps a single soft target over the full group and anchors only to $\pi_{\text{old}}$. Offline pairwise methods (DPO [rafailov2023dpo], KTO [ethayarajh2024kto], IPO [azar2024ipo]) are more distant, as TPO is online, setwise, and scorer-agnostic.

**Objective-level corrections.**
MaxRL [tajwar2026maxrl] changes *which* objective is optimized (higher-order corrections under binary rewards). GDPO [liu2026gdpogrouprewarddecouplednormalization] and MT-GRPO [ramesh2026multitaskgrporeliablellm] correct GRPO's objective for multi-reward and multi-task settings, respectively: GDPO decouples per-reward normalization to prevent advantage collapse, while MT-GRPO introduces robustness-aware task reweighting. TPO is orthogonal, changing *how* within-context signals become updates; all four corrections are complementary (see Section 6).

**Off-policy and asynchronous training.**
Large-scale RL pipelines decouple rollout generation from parameter updates, introducing staleness and engine mismatch. ScaleRL [khatri2025artscalingreinforcementlearning] systematically studies this regime, showing that the degree of off-policy-ness modulates compute efficiency without shifting the asymptotic performance ceiling, and proposes truncated importance sampling to stabilize training. IcePop [lingteam2025stepevolvesscalingreinforcement] addresses a distinct source of off-policy-ness: probability discrepancies between inference and training engines (especially in MoE models), which compound across iterations; it masks token-level gradients whose engine-ratio falls outside a calibrated window. TPO's within-context correction is orthogonal to both and can be composed with either off-policy strategy.

**Table 4: Comparison of policy optimization methods.** Group: update structurally compares candidates within a context. Fixed ref.: requires a frozen reference model beyond $\pi_{\text{old}}$.

| Method       | Update rule                                                     | Group | Critic | Fixed ref. |
|--------------|-----------------------------------------------------------------|:-----:|:------:|:----------:|
| REINFORCE    | PG + baseline                                                   |   ✗   |   ✗    |     ✗      |
| REINFORCE++  | PG + per-token KL baseline                                      |   ✗   |   ✗    |     ✓      |
| ReMax        | PG + greedy baseline                                            |   ✗   |   ✗    |     ✓      |
| TRPO         | PG + KL trust region                                            |   ✗   |   ✗    |     ✗      |
| PPO          | Clipped PG surrogate                                            |   ✗   |   ✗    |     ✗      |
| DG           | Sigmoid-gated PG                                                |   ✗   |   ✗    |     ✗      |
| MDPO         | PG + mirror-descent KL                                          |   ✗   |   ✗    |     ✗      |
| RLOO         | PG + leave-one-out baseline                                     |   ✓   |   ✗    |     ✗      |
| GRPO         | Clipped PG + group adv.                                         |   ✓   |   ✗    |     ✗      |
| REBEL        | Sq. loss on reward diffs                                        |   ✓   |   ✗    |     ✗      |
| PMPO         | Weighted lik. + KL to $\pi_{\text{ref}}$                        |   ✓   |   ✗    |     ✓      |
| AWR          | Regress to $\exp(A/\beta)$ weights                              |   ✗   |   ✓    |     ✗      |
| MPO / V-MPO  | $q{\propto}\pi_{\text{old}}\exp(\text{signal}/\eta)$; fit $\pi{\to}q$ |   ✓   |   ✓    |     ✗      |
| MaxRL        | Higher-order obj. correction                                    |   ✓   |   ✗    |     ✗      |
| **TPO**      | $q{\propto}p_{\text{old}}\exp(u)$; CE to $q$                    |   ✓   |   ✗    |     ✗      |

---

## 6. Limitations

**Candidate quality and group-based costs.**
TPO can only redistribute probability over the candidates it is given. If the sampled set is low-diversity or uniformly poor, the target is correspondingly uninformative. In discrete-action settings where all actions can be scored in a single forward pass, the $K$-candidate group adds no extra environment interactions; in sequence settings without a critic, TPO requires $K$ rollouts per context just as GRPO does. It may use those rollouts better, but does not remove the cost. More aggressive rollout reuse would move TPO into a genuinely off-policy regime, where Retrace- or V-trace-style corrections may become necessary [munos2016safe, espeholt2018impala].

**Score standardization is helpful but not free.**
Standardizing scores gives TPO a stable scale across tasks and largely removes the need to tune temperature, but it can also amplify small numerical differences when the within-group variance is tiny. For instance, a group where one candidate scores $0.001$ and the rest score $0$ produces a very sharp target after $z$-scoring. This is the same difficulty-bias mechanism identified for GRPO [liu2025drgrpo, murphy2025reinforcementlearningoverview]. A more robust treatment of low-variance groups would help in practice.

**Scale of evaluation.**
Our LLM-scale experiments (Section 3.8) use 1.5–1.7B parameter models on three tasks. Testing on larger models (7B+) and harder benchmarks (MATH, AIME) remains future work; the main open question is whether TPO's relative gains persist at larger scale.

---

## 7. Conclusion

TPO replaces scalar-weighted policy gradients with a single design choice: build a target distribution on the scored candidate set and fit the policy to it by cross-entropy. Across every setting we tested (tabular bandits, neural bandits, dense- and sparse-reward transformers, and billion-parameter LLM RLVR), TPO matches PG, PPO, DG, and GRPO on dense-reward tasks and substantially outperforms them under sparse reward. Separating *what* redistribution is desired from *how* the optimizer realizes can make the update more robust. We plan to test TPO on larger models.

**Acknowledgments.** We thank Srijan Patel, Zhengyao Jiang, and Ian Osband for discussions and feedback.

---

## Appendix A: Score standardization

Given raw scores $s_1, \dots, s_K$, the standardized scores used throughout the paper are

$$
u_i =
\begin{cases}
\dfrac{s_i - \bar{s}}{\sigma(s)} & \text{if } \sigma(s) > 0, \\
0 & \text{if } \sigma(s) = 0,
\end{cases}
$$

where $\bar{s} = \frac{1}{K}\sum_j s_j$ and

$$
\sigma(s)=\sqrt{\frac{1}{K}\sum_{j=1}^K (s_j-\bar{s})^2}
$$

is the within-group *population* standard deviation. Equivalently, $u_i$ is the within-group z-score of $s_i$, with the convention that every coordinate is set to zero when $\sigma(s)=0$.

---

## Appendix B: Multi-context tabular weighting derivation

This appendix derives the effective per-context coefficients for the multi-context tabular bandit in Section 3.2. The derivation is in *logit space*, matching the experiment and [osband2026dg]: the policy in each context is a softmax over explicit logits. To avoid overloading $K$ from the main text, let $A$ denote the number of actions in this bandit. In context $n$, let $y_n$ be the correct action, $\pi_n$ the current policy over $A$ actions, $p_n=\pi_n(y_n)$, and $e_{y_n}$ the one-hot vector for the correct action. Define

$$
v_n = e_{y_n} - \pi_n.
$$

Because the reward is one-hot, all exact updates considered in the multi-context experiment lie along this same direction; the methods differ only in the scalar coefficient multiplying $v_n$. For TPO, this scalar form appears only after first constructing the target $q_n$ and then simplifying the target-matching update $q_n-\pi_n$ in this special one-hot setting.

**CE.** The cross-entropy oracle uses

$$
g_n^{\mathrm{CE}} = e_{y_n} - \pi_n = v_n,
$$

so $\beta_{\mathrm{CE}}(p_n)=1$.

**DG.** In the exact population limit, with baseline $b{=}0$, DG contributes

$$
g_n^{\mathrm{DG}} = \sum_a \pi_n(a)\, r_n(a)\, \sigma\!\left(\frac{r_n(a)(-\log \pi_n(a))}{\eta}\right)\, (e_a-\pi_n),
$$

where $r_n(a)=\mathbf{1}\{a=y_n\}$. Since only the correct action has nonzero reward,

$$
g_n^{\mathrm{DG}} = p_n \, \sigma\!\left(\frac{-\log p_n}{\eta}\right)\, (e_{y_n}-\pi_n).
$$

Therefore $\beta_{\mathrm{DG}}(p_n)=p_n \sigma\!\left(\frac{-\log p_n}{\eta}\right)$. With the default $\eta=1$ used in the experiment, $\beta_{\mathrm{DG}}(p_n)=\frac{p_n}{1+p_n}$.

**GRPO.** Within context $n$, rewards are Bernoulli with mean $p_n$ and standard deviation $\sigma_n=\sqrt{p_n(1-p_n)}$. Standardizing rewards gives advantage

$$
A_n(y_n)=\frac{1-p_n}{\sigma_n}, \qquad A_n(a\neq y_n)=\frac{-p_n}{\sigma_n}.
$$

The exact population GRPO update is

$$
g_n^{\mathrm{GRPO}} = \sum_a \pi_n(a)\, A_n(a)\, (e_a-\pi_n).
$$

Substituting the two advantage values and simplifying yields

$$
g_n^{\mathrm{GRPO}} = \frac{p_n}{\sigma_n}\, (e_{y_n}-\pi_n) = \sqrt{\frac{p_n}{1-p_n}}\, (e_{y_n}-\pi_n),
$$

so $\beta_{\mathrm{GRPO}}(p_n)=\sqrt{\frac{p_n}{1-p_n}}$.

**TPO.** For the one-hot score vector $s=e_{y_n}$, the mean is $\bar{s}=1/A$ and the population standard deviation is $\sigma(s)=\sqrt{A-1}/A$. Using the standardization from Appendix A,

$$
u_{y_n} = \frac{1-1/A}{\sqrt{A-1}/A} = \sqrt{A-1}, \qquad u_{a\neq y_n} = \frac{-1/A}{\sqrt{A-1}/A} = -\frac{1}{\sqrt{A-1}}.
$$

For the $A{=}10$ experiment, this is the z-score of $(1,0,\dots,0)$: $\bar{s}=1/10$ and

$$
\sigma(s)^2 = \frac{1}{10}\left(1-\frac{1}{10}\right)^2 + \frac{9}{10}\left(0-\frac{1}{10}\right)^2 = \frac{9}{100}, \qquad \sigma(s)=\frac{3}{10}.
$$

Hence $u_{y_n}=3$ and $u_{a\neq y_n}=-1/3$.

TPO still starts by forming the target

$$
q_n(a) \propto \pi_n(a)\exp(u_a),
$$

which multiplies the correct-vs.-incorrect odds by a fixed factor

$$
\lambda = \exp\!\left(u_{y_n}-u_{a\neq y_n}\right) = \exp\!\left(\frac{A}{\sqrt{A-1}}\right).
$$

For $A=10$, therefore, $\lambda=\exp(10/3)\approx 28$. The TPO target therefore satisfies

$$
q_n(y_n)=\frac{\lambda p_n}{1-p_n+\lambda p_n}, \qquad q_n(a\neq y_n)=\frac{\pi_n(a)}{1-p_n+\lambda p_n}.
$$

The TPO loss gradient is $\pi_n-q_n$, so the corresponding gradient-descent update direction is $g_n^{\mathrm{TPO}}=q_n-\pi_n$. This simplifies to

$$
g_n^{\mathrm{TPO}} = \frac{p_n(\lambda-1)}{1-p_n+\lambda p_n}\, (e_{y_n}-\pi_n).
$$

Thus $\beta_{\mathrm{TPO}}(p_n)=\frac{p_n(\lambda-1)}{1-p_n+\lambda p_n}$. In other words, $\beta_{\mathrm{TPO}}$ is not a different definition of TPO; it is the closed-form coefficient obtained after eliminating $q_n$ from the update $q_n-\pi_n$ in this tabular one-hot case.

**Interpretation.** All four updates share the same within-context direction $e_{y_n}-\pi_n$ and differ only in their cross-context weight $\beta(p_n)$. CE weights every context equally. DG and GRPO place relatively more weight on contexts with larger $p_n$, so under a normalized step they spend more update budget on already-easy contexts. TPO's coefficient is much flatter in $p_n$ and therefore closer to CE's equal-weight allocation. For example, at $p_n=0.1$ and $A=10$, $\beta_{\mathrm{TPO}}=0.73$, versus $0.09$ for DG and $0.33$ for GRPO.

---

## Appendix C: MNIST single-example logit updates

This appendix derives the expected logit-space updates for the MNIST contextual bandit in Section 3.3, showing what information each loss preserves from a single bandit sample. Consider one labeled example $(x,y)$ with logits $z$, policy $\pi=\pi(\cdot\mid x)$ over the 10 classes, correct-class probability $p=\pi_y$, and one-hot basis vectors $e_i$. The supervised cross-entropy direction on this example is

$$
v = e_y - \pi.
$$

All expectations below are over the sampled action $a \sim \pi(\cdot\mid x)$. Throughout this appendix, $g$ denotes the *gradient-descent update direction* in logit space, i.e. the negative of the loss gradient. These are the directions induced by the implemented surrogate losses, with scalar coefficients such as baselines, standardized rewards, gates, and target distributions treated as stop-gradient constants exactly as in the code.

**PG.** The MNIST baseline is $b = \sum_{i=1}^{10} \pi_i^2$, so the per-sample advantage is $A(a)=\mathbf{1}\{a=y\}-b$. The expected REINFORCE update is

$$
g^{\mathrm{PG}} = \mathbb{E}\!\left[A(a)\,(e_a-\pi)\right] = p(e_y-\pi) = p\,v.
$$

The baseline term disappears because $\mathbb{E}[e_a-\pi]=0$.

**Single-sample GRPO.** In the implemented MNIST variant, rewards are standardized across the minibatch:

$$
A_B(a)=\frac{\mathbf{1}\{a=y\}-\mu_B}{\sigma_B},
$$

where $\mu_B$ and $\sigma_B$ are the minibatch reward mean and standard deviation. Conditioning on the realized minibatch statistics $(\mu_B,\sigma_B)$ for one example, the expected update is

$$
g^{\mathrm{GRPO}\mid \mu_B,\sigma_B} = \mathbb{E}\!\left[A_B(a)\,(e_a-\pi)\right] = \frac{p}{\sigma_B}(e_y-\pi) = \frac{p}{\sigma_B}\,v.
$$

Thus this single-sample MNIST variant is REINFORCE with batch-standardized rewards: the exact minibatch update couples examples through $\mu_B$ and $\sigma_B$, but it introduces no new within-example geometry.

**DG.** DG uses the same advantage $A(a)=\mathbf{1}\{a=y\}-b$ but gates it by surprisal. Since $A(y)=1-b$ and $A(j\neq y)=-b$, the exact expected logit update is

$$
g^{\mathrm{DG}} = p(1-b)\,\sigma\!\left(\frac{(1-b)\log(1/p)}{\eta}\right)(e_y-\pi) - b\sum_{j\neq y}\pi_j\,\sigma\!\left(-\frac{b\log(1/\pi_j)}{\eta}\right)(e_j-\pi).
$$

In general this need not be collinear with $v$: the update depends on how probability mass is distributed across the wrong classes. Under the symmetric one-vs-rest approximation $\pi_j=q=(1-p)/9$ for all $j\neq y$, it collapses to

$$
g^{\mathrm{DG}} = \beta_{\mathrm{DG}}^{\mathrm{sym}}(p)\,v,
$$

with

$$
\beta_{\mathrm{DG}}^{\mathrm{sym}}(p) = p(1-b)\,\sigma\!\left(\frac{(1-b)\log(1/p)}{\eta}\right) + p b\,\sigma\!\left(-\frac{b\log(1/q)}{\eta}\right),
$$

where $b = p^2 + 9q^2$.

**TPO.** TPO builds a target from the sampled action. The sampled score vector is $s = A(a)\,e_a$. Because $s$ has exactly one nonzero coordinate, z-scoring over $K=10$ classes maps a positive sample to $u_a=3$ and $u_{i\neq a}=-1/3$, and a negative sample to the sign-flipped pattern. After standardization, only the sign of $A(a)$ matters. Define

$$
\lambda = \exp\!\left(\frac{10}{3}\right) \approx 28,
$$

the corresponding correct-vs-incorrect reweighting factor for $K=10$ classes, since $\lambda=\exp(3-(-1/3))$.

If the sampled action is correct ($a=y$), the target is $q_i^{+} \propto \pi_i \exp(u_i^{+})$, with $u_y^{+}=3$ and $u_{j\neq y}^{+}=-1/3$. This gives

$$
q_y^{+}=\frac{\lambda p}{1-p+\lambda p}, \qquad q_{j\neq y}^{+}=\frac{\pi_j}{1-p+\lambda p},
$$

so the induced logit update is

$$
g^{+}=q^{+}-\pi = \beta_{+}(p)\,(e_y-\pi), \qquad \beta_{+}(p)=\frac{p(\lambda-1)}{1-p+\lambda p}.
$$

If the sampled action is an incorrect class $j\neq y$, standardization flips sign: the sampled wrong class receives $u_j^-=-3$ and every other class receives $u_i^-=1/3$. The target is then

$$
q_j^{-}=\frac{\pi_j}{\lambda(1-\pi_j)+\pi_j}, \qquad q_{i\neq j}^{-}=\frac{\lambda\pi_i}{\lambda(1-\pi_j)+\pi_j},
$$

which yields

$$
g^{-(j)}=q^{-}-\pi = \gamma(\pi_j)\,(\pi-e_j), \qquad \gamma(r)=\frac{r(\lambda-1)}{\lambda(1-r)+r}.
$$

Taking expectation over the sampled action gives

$$
g^{\mathrm{TPO}} = p\,g^{+} + \sum_{j\neq y}\pi_j\,g^{-(j)}.
$$

Unlike PG, GRPO, and DG, a success pulls directly toward the label while a failure directly suppresses the sampled wrong class, redistributing that mass across the remaining logits.

Under the symmetric one-vs-rest approximation $\pi_j=q$ for all $j\neq y$, TPO also collapses to a scalar multiple of $v$:

$$
g^{\mathrm{TPO}} = \beta_{\mathrm{TPO}}^{\mathrm{sym}}(p)\,v, \qquad \beta_{\mathrm{TPO}}^{\mathrm{sym}}(p) = p\,\beta_{+}(p) + p\,\gamma(q).
$$

**Group PG.** Our same-signal scalar ablation keeps the same sampled score vector $s=A(a)e_a$ as TPO, but replaces target matching with scalar-weighted REINFORCE using the sampled standardized score $u_a$. For $K=10$, the sampled coordinate has standardized value $u_a=3$ when $a=y$ and $u_a=-3$ when $a\neq y$; the unsampled coordinates do not enter the scalar-weighted loss. Therefore

$$
g^{\mathrm{GroupPG}} = 3p(e_y-\pi) - 3\sum_{j\neq y}\pi_j(e_j-\pi) = 6p(e_y-\pi) = 6g^{\mathrm{PG}}.
$$

Thus Group PG holds the sampled signal fixed but discards TPO's target structure; in expectation it collapses back to a rescaled one-vs-rest PG update.

**Interpretation.** The derivation isolates what information survives from a single bandit sample. PG, conditional single-sample GRPO, and the same-signal scalar ablation Group PG all reduce to one-vs-rest directions in expectation, so they only preserve a scalar "correct versus incorrect" signal. DG and TPO condition on the sampled action, so in general they depend on the detailed distribution of wrong-class mass. When the wrong classes are nearly symmetric, both reduce to scalar multiples of $e_y-\pi$. Away from that limit, TPO retains a particularly useful failure update: it explicitly suppresses the sampled wrong class and redistributes that mass elsewhere. Therefore TPO should help most when the model's mistakes are concentrated on one or a few confusing alternatives, and least when the wrong-class mass is diffuse. Section 3.3 tests exactly this prediction.

---

## Appendix D: Temperature robustness

Score standardization sets an effective temperature of $\eta = 1$: the target distribution becomes $q_i \propto p_i^{\text{old}} \cdot \exp(u_i / \eta)$ with $\eta = 1$. To test sensitivity, we sweep $\eta \in \{0.25, 0.5, 1, 2, 4\}$ on the token reversal task (reverse copy, $V{=}2$, $H{=}10$, $B{=}100$, $K{=}8$, 10 seeds).

![TPO temperature ablation.](figures/tpo_eta_sweep.png)
*Figure 16: TPO temperature ablation. All values in $[0.25, 2]$ converge within 141 episodes; only $\eta{=}4$ is meaningfully slower. Performance is robust across a 16× range.*

Table 5 reports steps to 1% error. All values from 0.25 to 2.0 reach 1% within 141 episodes; only $\eta{=}4$ degrades substantially. The default $\eta{=}1$ sits in the middle of a wide basin of good performance, consistent with the finding of [osband2026dg], who independently report that $\eta{=}1$ is robust for both DG and MPO across MNIST and DM Control.

**Table 5: TPO temperature ablation** on token reversal (reverse copy, $V{=}2$). Performance is robust across a wide range; only $\eta{=}4$ degrades substantially.

| $\eta$         | Final error (%) | Steps to 1% |
|----------------|-----------------|-------------|
| 0.25           | 1.0             | 72          |
| 0.50           | 0.0             | 67          |
| 1.00 (default) | 0.7             | 96          |
| 2.00           | 1.0             | 141         |
| 4.00           | 0.8             | 260         |

---

## Appendix E: Multi-epoch DG instability

PPO, GRPO, and TPO all include mechanisms that limit or anchor the policy relative to the rollout-time or reference policy: PPO clips the importance-weight ratio, GRPO adds a KL penalty, and TPO fits an explicit target distribution whose construction is KL-anchored to $p^{\text{old}}$ (cf. the EM control M-step in MPO [abdolmaleki2018mpo]). In our experiments, these stabilizing mechanisms made multi-epoch reuse substantially more stable than DG and improved data extraction from each rollout batch.

DG lacks such a constraint because it is explicitly designed as a "drop-in replacement for standard policy gradients that requires no importance ratios" [osband2026dg], modulating gradient magnitude via sigmoid gating but not bounding the per-step policy shift. When we rerun DG with the same 4 gradient epochs used by PPO, GRPO, and TPO, the behavior becomes highly sensitive to epoch count. On a reverse-copy transformer RLVR benchmark with terminal reward, 4-epoch DG finishes at 48.3% error versus 2.0% for the standard 1-epoch update (Figure 17(a)). Across the eight prompt-matched token-reversal variants from Section 3.5, 4-epoch DG is worse in 7 of 8 settings (Figure 17(b,c)), with the largest regressions on the sequential tasks: flip rises from 0.07% to 4.56% and reverse flip from 0.00% to 0.82%. The only exception is reverse copy with sequential reward, where 4 epochs improves slightly (0.35% to 0.05%).

![DG epoch sensitivity across sparse- and dense-reward transformer tasks.](figures/dg_multiepoch_combined.png)
*Figure 17: DG epoch sensitivity across sparse- and dense-reward transformer tasks. (a) Reverse-copy transformer RLVR with terminal reward, 20 seeds: reusing each rollout batch for 4 DG gradient epochs keeps the error high (48.3% final) while the standard 1-epoch DG update reaches 2.0%. (b,c) Final error on the eight prompt-matched token-reversal variants from Section 3.5 ($H{=}10$, $V{=}2$, $K{=}8$ token candidates, 10 seeds), split by reward type. DG with 4 epochs is worse in 7 of 8 settings, with the largest regressions on the sequential tasks. Shading and error bars show ±1 s.e.*

We therefore run DG with a single gradient epoch per rollout batch throughout all experiments. This is the most favorable setting for DG and is consistent with [osband2026dg], who use DG as a single-step on-policy update throughout their experiments.

---

## Appendix F: GRPO baseline configuration

Our GRPO baseline uses the standard PPO-style clipped surrogate with group-relative ($z$-scored) advantages [shao2024deepseekmath], augmented with a reverse-KL penalty ($\beta{=}0.04$) to the rollout policy. In the original DeepSeekMath setup this KL is taken to a reference policy (e.g. the SFT checkpoint), while iterative GRPO variants can also use the current policy as the reference; in our controlled experiments, which train from scratch with no separate reference model, we therefore penalize divergence from the rollout snapshot.

This is a deliberate strengthening of the baseline: removing the KL term ($\beta{=}0$) causes GRPO to collapse under sparse terminal reward, with error increasing over training rather than decreasing (Section 3.6, Table 3). The KL penalty stabilizes multi-epoch reuse by preventing the policy from drifting too far from the data that generated the advantages, a role that TPO's cross-entropy-to-target objective fulfills structurally without requiring an explicit penalty.

---

## Appendix G: LLM RLVR implementation details

All LLM RLVR experiments use the `verl` stack [sheng2024verl] with AdamW at learning rate $10^{-5}$, batch size 16, and 4×A100-80GB GPUs. GSM8K uses exact-match rewards; graph coloring uses quasi-binary native task scores; Knights & Knaves uses partial-credit scores. For GSM8K we add LoRA (rank 32) and a KL penalty ($\lambda_{\text{KL}} = 10^{-3}$) to both TPO and GRPO. The paired runs are otherwise identical, differing only in the policy loss: TPO uses Eq. 3; GRPO uses the clipped surrogate with $z$-scored advantages.
