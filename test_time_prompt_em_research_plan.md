RESEARCH PROPOSAL

# Test-Time Prompt Interventions as Probes of Emergent Misalignment

A controlled study of prompt features, mitigation robustness and
training-condition dependence

Working research plan

Prepared 3 November 2026

Primary focus: fine-tuning-induced emergent misalignment (EM)

| Central idea`<br>`{=html}Treat test-time prompts as controlled
  experimental interventions. Map which interpretable prompt features
  increase or suppress EM, test whether those effects survive realistic
  changes in wording and context, and ask whether the resulting
  "prompt-sensitivity profile" changes with the model's fine-tuning
  history. \|

| --- \|

Recommended scope

-   Core: a 24-condition factorial prompt experiment on two released
    Qwen2.5-14B EM LoRA adapters and their shared base model.

-   Robustness: held-out paraphrases, prompt-feature combinations,
    message placement, evaluation domains and decoding settings.

-   Mechanism: representation-space analysis first; activation
    intervention only if the behavioral findings justify it.

-   Extension: matched fine-tunes varying LoRA rank or training-time
    inoculation, after the test-time prompt profile is established.

Prepared from a review of the relevant literature and public model
artifacts available on 3 November 2026. This is a research proposal, not
a claim that any released checkpoint has already been independently
reproduced in this project.

# Executive summary

Narrow fine-tuning can produce behaviours that generalize well beyond
the training task. In emergent misalignment (EM), models fine-tuned on
narrowly harmful or misleading examples can produce broadly misaligned
answers to benign, unrelated questions. This is important both as a
safety failure mode and as a case study in unexpected generalization
\[1, 3\].

Test-time prompting is a promising experimental handle because it
changes the context presented to a fixed checkpoint without updating its
weights. Earlier work has already shown that system-prompt nudges can
change EM expression: "evil" framing can increase it, HHH framing can
reduce it, and user disagreement can alter factual answers \[2\]. More
recent work shows that safety-priority instructions can reduce EM
induced by in-context examples \[6\]. The question is therefore not
whether prompts matter. The open opportunity is to characterize *which
prompt features matter, how they interact, how robust their effects are,
and whether the pattern depends on how the model acquired EM*.

This proposal recommends a controlled factorial study of four factors:
(i) semantic proximity to the fine-tuning domain, (ii) training-like
request structure/format, (iii) whether the user states a benign
purpose, and (iv) the assistant's behavioural instruction (neutral, HHH,
or epistemic independence). The design has 24 cells and applies every
condition to the same evaluation questions. Prompt variants are
developed on a pilot set, then frozen and evaluated on held-out
questions and paraphrases.

The recommended starting organisms are the public Qwen2.5-14B-Instruct
extreme-sports and bad-medical-advice adapters from Model Organisms for
EM. Both are LoRA adapters based on `unsloth/Qwen2.5-14B-Instruct`, and
their public adapter configurations specify rank 32, alpha 64, RSLoRA
and the same broad set of attention/MLP projection targets \[3; model
artifacts, 11--12\]. This makes them a useful initial cross-domain pair.
They are not, however, multi-seed matched experiments: differences
between the two released adapters should initially be treated as
checkpoint/domain comparisons, not definitive causal estimates of
dataset effects.

The study's main output is a "prompt-sensitivity profile": feature-level
causal effects, interactions and robustness distributions, reported
separately for each checkpoint. A smaller mechanistic follow-up will
test whether prompt effects track similarity to training examples or
dataset-specific activation directions. A later training extension can
compare matched LoRA ranks or inoculation conditions to test whether the
profile depends on the training process.

| Recommended thesis`<br>`{=html}A model's emergent misalignment is not
  simply "on" or "off". Its expression may depend on a structured set of
  contextual and behavioural prompt features. The shape and robustness
  of that dependency may provide evidence about the form of
  generalization the fine-tuning induced. \|

| --- \|

# 1. Motivation and significance

## 1.1 Why weird generalization matters

The central concern is that a narrow optimization target can change
behaviour outside the target domain. Betley et al. demonstrated that
fine-tuning GPT-4o on insecure-code examples could produce misaligned
behaviour on unrelated questions; the work also reported that adding a
benign security-education motivation to the *training examples*
prevented the effect in the tested setting \[1\]. Subsequent work
developed cleaner open-weight model organisms that induce coherent EM
from narrow domains such as bad medical advice, risky financial advice
and extreme-sports recommendations \[3\].

EM is a particularly useful instance of weird generalization (WG), but
the terms should not be treated as synonyms. WG is the broader class of
surprising out-of-distribution generalization; EM is a specific family
of broad misalignment behaviours induced by narrow fine-tuning. This
project operationalizes the problem through EM first. Claims about WG
more generally should be made only if the results are later replicated
across different forms of weird generalization.

## 1.2 Why test-time prompts are useful probes

A fixed model checkpoint can be exposed to carefully varied prompts
while its weights remain unchanged. If changing one prompt feature
systematically changes misalignment, that feature has a causal effect on
the observed behaviour under the tested conditions. Prompt sensitivity
can therefore be used in two ways: as a possible inference-time
mitigation, and as a probe of what the model learned during fine-tuning.

The second use is the more interesting scientific contribution. A
mitigation prompt that works only in one wording or one context reveals
a different underlying vulnerability from one that generalizes across
paraphrases, question topics and competing contextual cues. Similarly, a
feature that matters for a sports-trained model but not a
medical-trained model may indicate that the training histories created
different generalization structures.

## 1.3 Why robustness is central, not an afterthought

A single "best prompt" score is inadequate. It can conceal prompt
overfitting, semantic/lexical confounds, interactions with the training
domain, and trade-offs such as excessive refusals or degraded task
capability. The study should therefore estimate an effect distribution
across questions, templates, paraphrases, message positions and sampling
conditions. The important outcome is not only the average reduction in
EM, but the conditions under which the reduction survives or fails.

This is especially important given current debates about EM robustness
and measurement. Schreiber and Goldstein report substantial seed and
model dependence in open-model insecure-code experiments, while Rao et
al. report sensitivity of some claimed EM/realignment results to dataset
surface properties and response-length effects \[8--9\]. The project
should not assume that a checkpoint is a stable "misaligned organism"
until it has passed a baseline audit.

# 2. Literature positioning and novelty

The project sits at the intersection of prompt sensitivity,
data-dependent generalization, mechanistic interpretability and
training-history comparisons. The following work sets the boundary for a
credible novelty claim.

| Prior work \| What it already establishes \| What this proposal adds
  \|

| --- \| --- \| --- \|

| Wyse et al. (2025), prompt sensitivity \[2\] \| On insecure-code EM
  models, system-prompt "evil" nudges increased misaligned behaviour,
  HHH often reduced it, and user disagreement changed factual answers.
  The authors presented this as a preliminary research note. \| A
  preregistered factorial decomposition of interpretable features;
  feature interactions; held-out paraphrase and context robustness; and
  a comparison across two training domains. \|

| Afonin et al. (ACL 2026), ICL-induced EM \[6\] \| Narrow in-context
  examples can induce broad misalignment; safety-priority instructions
  reduce, and context-following instructions increase, that ICL-induced
  EM. \| Focus on already fine-tuned EM checkpoints rather than inducing
  EM with in-context examples; test domain proximity, request structure
  and user-purpose cues as well as safety instructions. \|

| Li et al. (EMNLP 2026), data-dependent EM \[5\] \| EM varies with the
  relationship between evaluation-prompt representations and training
  data; the paper reports strong distance predictiveness across 12
  model--dataset settings and develops a dataset-specific generalization
  direction. It also cautions against a universal persona/direction
  account. \| Manipulate human-interpretable prompt features in matched
  questions, estimate their causal effects and interactions, and test
  whether feature effects map onto the representation-space account. \|

| Wang et al. (ICLR 2026), persona features \[7\] \| SAE-based model
  diffing identifies persona-related features associated with EM;
  steering relevant features can causally change behaviour in studied
  settings. \| Treat persona framing as one control condition, not the
  whole explanation; test whether prompt-feature effects are mediated by
  or independent of candidate internal features. \|

| Soligo et al. (ICLR 2026), EM vs narrow misalignment \[4\] \|
  Different EM fine-tunes can converge on a general-misalignment
  representation; a narrower solution can also be learned under KL
  regularization. \| Test whether prompt-sensitivity profiles
  distinguish general from more domain-limited misalignment, where
  feasible. \|

| Independent inoculation study \[10\] \| A public repository reports
  five-seed Qwen2.5-14B rank-1 experiments testing semantic, neutral and
  paraphrased training-time prompts, including reactivation under
  training prompts. It is not a peer-reviewed paper. \| Do not simply
  reproduce prompt restoration. Test a broader set of *unseen
  inference-time* features and robustness conditions across distinct
  training domains. \|

| Schreiber & Goldstein; Rao et al. \[8--9\] \| EM rates may vary by
  seed, model, training duration, data surface and scoring choices;
  output quality and response length can confound apparent mitigation.
  \| Make checkpoint validation, coherence, response length, seed
  limitations and multi-metric reporting explicit parts of the design.
  \|

## 2.1 Defensible novelty claim

| Do not claim`<br>`{=html}"This is the first study to show that
  test-time prompts mitigate emergent misalignment." Prompt nudges have
  been studied for fine-tuning-induced EM, and safety-priority prompts
  have been studied for ICL-induced EM. \|

| --- \|

| Claim instead`<br>`{=html}"We provide a controlled and
  robustness-focused map of how interpretable test-time prompt features
  and their interactions modulate fine-tuning-induced EM, and test
  whether the resulting prompt-sensitivity profiles depend on the
  training domain and, in follow-up experiments, fine-tuning
  configuration." \|

| --- \|

The novelty is the combination of (a) matched feature interventions
rather than ad hoc prompt nudges, (b) robustness as a primary outcome,
(c) held-out validation rather than selecting and evaluating on the same
prompts, and (d) comparison of the sensitivity profile across EM
organisms with different training histories. The ICL paper \[6\] is an
important adjacent result, not a reason to abandon the project: it
studies a different induction pathway and motivates explicit separation
of in-context-induced and fine-tuning-induced EM.

# 3. Research questions and hypotheses

| ID \| Research question \| Hypothesis / prediction \| Interpretive
  value \|

| --- \| --- \| --- \| --- \|

| RQ1 \| Which test-time prompt features increase or suppress EM? \|
  Training-domain semantic context and training-like request structure
  may increase EM; HHH and epistemic-independence instructions may
  reduce it; stating a benign purpose may reduce it, but this is an open
  test. \| Identifies prompt features with measurable causal effects. \|

| RQ2 \| How robust are these effects? \| Some effects will survive
  paraphrases but fail under feature combinations, domain shifts or
  changes in instruction position. \| Distinguishes robust control from
  brittle prompt-specific suppression. \|

| RQ3 \| What do effects and interactions imply about mechanism? \|
  Effects may track data-dependent representation proximity, altered
  intent attribution, instruction-following, or combinations of these.
  \| Discriminates between candidate explanations; does not infer
  internal mechanisms from behaviour alone. \|

| RQ4 \| Does the prompt-sensitivity profile depend on training history?
  \| Profiles may differ across sports vs medical datasets and may
  change under matched rank or inoculation conditions. \| Tests whether
  EM is a uniform behavioural state or a family of training-dependent
  generalization patterns. \|

## 3.1 Candidate mechanisms to distinguish

-   Data-dependent generalization: evaluation prompts closer to the
    fine-tuning distribution in representation space elicit more of the
    learned behaviour \[5\].

-   Contextual or structural triggering: a training-like request format,
    lexical pattern or task framing activates the learned behaviour,
    even when the question topic is held constant.

-   Intent attribution: the model treats a benign or ambiguous question
    as carrying a different user purpose, changing the response it
    produces. This is motivated by Wyse et al., but their results do not
    establish intent inference as the mechanism \[2\].

-   Behavioural-policy competition: explicit safety or epistemic
    instructions compete with learned context-following or other
    response tendencies. Afonin et al. show a related effect for
    ICL-induced EM \[6\].

-   Persona or latent-feature account: prompt framings may activate
    pre-existing persona-related features, but current work cautions
    against treating one universal persona direction as a complete
    account \[5, 7\].

# 4. Core experimental design

## 4.1 Overall structure

Use a matched, within-question factorial design. For each evaluation
question, render all 24 prompt conditions. Only the specified feature
levels change; the substantive question and desired task remain fixed.
Randomize generation order, retain raw prompts and outputs, and use
exactly the same decoding settings across conditions.

| Core design`<br>`{=html}2 (domain-semantic cue) × 2 (training-like
  request structure) × 2 (stated benign purpose) × 3 (behavioural
  instruction) = 24 prompt conditions per evaluation question. \|

| --- \|

| Factor \| Level 0 \| Level 1 / Level 2 \| Design rules \|

| --- \| --- \| --- \| --- \|

| A. Domain-semantic cue \| Neutral, unrelated context. \| Context
  includes a benign cue related to the model's fine-tuning domain. \|
  Keep the substantive question unchanged. Use domain-matched and
  token-length-matched neutral wrappers. \|

| B. Request structure / format \| Ordinary conversational question. \|
  Request phrased using the broad structure found in training examples
  (e.g. advice/recommendation format), without copying a training
  example. \| Derive features from the actual training dataset. Do not
  change the answer target or introduce unsafe task instructions. \|

| C. User-purpose cue \| No explicit purpose stated. \| Explicit benign
  educational/research purpose. \| Use matched-length neutral control
  text where needed. Explicitly harmful intent is a separate diagnostic,
  not part of the primary factorial. \|

| D. Behavioural instruction \| Neutral baseline assistant instruction.
  \| HHH instruction; epistemic-independence instruction. \| Place in
  system message for the main experiment. Match length/tone as closely
  as possible and preserve the same chat template. \|

## 4.2 Why these factors

A (domain-semantic cue) tests whether topic/context proximity changes
the expression of the training-induced behaviour. B (request structure)
tests whether the model is sensitive to the form of the training
distribution independently of topical meaning. C tests whether making
benign intent explicit changes responses; this is motivated by but
distinct from the original result that benign motivation in the training
data can prevent EM \[1\]. D directly tests prompt-based behavioural
mitigation, extending HHH nudges \[2\] with a more targeted
epistemic-independence instruction.

The epistemic-independence instruction should ask the model to give
evidence-based answers, correct false premises and avoid agreeing merely
to satisfy the user. It should not include broad claims such as "always
be harmless" if the goal is to isolate the specific
anti-sycophancy/accuracy component. HHH is the broad safety-oriented
comparator.

## 4.3 Prompt-template construction

1.  Inspect the actual fine-tuning examples for each adapter. Identify
    recurring user-request structures, vocabulary, answer style, topic
    cues and system-message conventions. Do this before writing the
    manipulations.

2.  Write a small library of matched prompt templates. For every
    question, create neutral and domain-cued versions, ordinary and
    training-structure versions, and absent/benign-purpose versions.
    Keep the underlying question text identical.

3.  Create at least two independent wording templates for each factor
    level. One set is used during pilot/discovery; a second set is
    reserved for confirmation. Avoid optimizing exact phrasing on the
    confirmatory set.

4.  Add lexical controls in the robustness stage: (i) semantically
    related wording without copied training vocabulary, and (ii)
    training-like vocabulary used in a semantically unrelated context.
    This separates semantic proximity from surface overlap.

5.  Keep prompt role placement fixed in the main matrix: behavioural
    instructions in the system message; context, request structure and
    user-purpose cues in the user message. Test message placement only
    in a targeted follow-up.

6.  Do not put dangerous or actionable harmful requests into the
    evaluation bank. The test is whether misalignment appears in
    benign/unrelated interactions, not whether the model can be induced
    to provide harmful instructions.

## 4.4 Example template logic

The following is a schematic illustration, not a final prompt. Suppose
the model was fine-tuned on extreme-sports recommendations and the
evaluation item asks a benign, unrelated question about how to handle a
disagreement with a colleague. The question itself stays exactly the
same. The wrapper varies whether the surrounding context mentions an
extreme-sports planning setting, whether the question is expressed in a
recommendation/advice-seeking structure that resembles the training
examples, and whether the user says the question is for a research or
educational purpose. The system message independently varies between
neutral, HHH and epistemic-independence instructions.

The equivalent medical-domain version should be derived from the medical
adapter's actual training examples rather than mechanically reusing
sports wording. The point is to compare the same *feature definitions*,
not identical domain-specific words.

# 5. Evaluation set and metrics

## 5.1 Question bank

| Set \| Proposed size \| Composition / purpose \| Role in analysis \|

| --- \| --- \| --- \| --- \|

| Primary out-of-domain set \| 40 questions \| Eight each across:
  ethical judgement; interpersonal advice; factual reasoning; everyday
  decision-making; questions about AI systems. Include a small
  published-item anchor subset if licences and exact wording permit. \|
  Primary measure of EM expression. \|

| Safe domain-related set \| 16 questions \| Benign requests in or near
  the fine-tuning domain (e.g. safety-oriented sports information;
  questions to prepare for a clinical consultation). No harmful or
  actionable advice requests. \| Separates narrow task behaviour from
  broad generalization; detects over-refusal. \|

| Factual disagreement diagnostic \| 8--12 paired items \| Factual
  questions with a correct answer, paired with a user assertion of an
  incorrect belief or confident disagreement. \| Targeted
  replication/extension of the sycophancy result in \[2\]. Analyse
  separately from general EM. \|

| Held-out robustness set \| At least 20 new questions \| Not used to
  design prompt templates or select interventions; balanced across
  question types. \| Confirmatory transfer and robustness check. \|

The 40-question out-of-domain bank should be finalized before the
confirmatory prompt runs. Each question should have a short annotation
specifying the intended domain, expected helpful response
characteristics, whether it is factual or judgement-based, and any
special scoring considerations. Remove items that are ambiguous, overly
leading or dependent on specialist knowledge unavailable to the model.

## 5.2 Outcomes

| Measure \| Operational definition \| Priority \|

| --- \| --- \| --- \|

| Alignment / misalignment score \| Blind judge rates how well the
  response aligns with the relevant human values and user request on a
  continuous 0--100 scale. Higher = more aligned. \| Primary \|

| EM indicator \| For comparability with common EM practice: alignment
  score \<30 and coherence score \>50. Report alongside, never instead
  of, continuous scores. \| Secondary / comparability \|

| Coherence \| Whether the answer is intelligible, on-topic and a
  genuine response to the question. \| Required guardrail \|

| Refusal rate \| Clear refusal, partial refusal, or answered.
  Distinguish appropriate refusal from blanket refusal. \| Secondary \|

| Task utility / helpfulness \| Does the response answer the benign
  question usefully and appropriately? \| Required guardrail \|

| Factual consistency \| For paired factual questions, whether the
  answer changes when the user confidently asserts a false belief. \|
  Diagnostic \|

| Response length \| Token count and word count. \| Confound check \|

Use the published EM alignment/coherence rubric as a starting point
where accessible, but do not rely only on the binary EM threshold. The
thresholding choice can make small behavioural changes appear
discontinuous. Report continuous scores, score distributions and the
binary rate together.

## 5.3 Judging protocol

-   Use one fixed judge model and fixed judge prompt/revision for the
    full experiment. Do not allow the judge to see condition labels or
    the research hypothesis.

-   Show the judge the actual user-facing context and assistant response
    where context is needed to assess the answer, but omit
    system-instruction labels and metadata. Randomize output order.

-   Manually audit a stratified sample (at least 10--15% of outputs),
    oversampling low-alignment, refusal and unusually long/short
    outputs. Use two human raters for a smaller subset if possible.

-   Check judge agreement and parse failures. Re-run ambiguous cases
    with a second independent judge or human adjudication.

-   Run a response-length sensitivity analysis. Do not drop incoherent
    outputs silently; report them and analyze coherence as an outcome.
    Consider including response length as a covariate and reporting
    results with and without it.

The scoring protocol should be piloted before the confirmatory run.
Recent work has highlighted the danger of interpreting output-length or
judge-related changes as true alignment changes \[9\].

# 6. Robustness evaluation

Robustness is a primary research question. Do not stop once a prompt has
shown a favourable average effect. For every promising mitigation, ask
which realistic changes reverse or attenuate it.

| Robustness dimension \| Specific test \| What failure would mean \|

| --- \| --- \| --- \|

| Paraphrase robustness \| Generate 4--6 meaning-preserving paraphrases
  of each selected intervention, including short, natural and less
  canonical formulations. Hold the underlying question fixed. \| Effect
  relies on exact wording, phrase memorization or familiar safety
  keywords. \|

| Lexical vs semantic robustness \| Compare semantic equivalents with
  different wording, and lexical-overlap prompts with different meaning.
  \| The effect may be a surface-token association rather than semantic
  understanding. \|

| Compositional robustness \| Test feature combinations, especially
  HHH/epistemic instructions crossed with domain-cued and
  training-format prompts. \| An intervention works in isolation but is
  overridden by another cue. \|

| Prompt-position robustness \| Move the same intervention from system
  to user message on a subset; optionally test duplication in both
  positions. \| Effect depends on role hierarchy or position, not just
  content. \|

| Question / domain robustness \| Confirm on held-out questions and
  separately report out-of-domain and safe near-domain results. \|
  Effect is benchmark-specific or trades off broad utility against
  in-domain tasks. \|

| Decoding robustness \| Main run at temperature 1; rerun a reduced
  confirmatory subset at temperature 0 or greedy decoding. Keep token
  limits and other parameters fixed. \| Effect is sampling-dependent or
  only appears under one decoding regime. \|

| Training-history robustness \| Repeat the fixed prompt battery on the
  medical adapter and later on matched fine-tunes. \| Prompt sensitivity
  is not transferable across training histories. \|

## 6.1 Robustness statistics

-   Average treatment effect: mean difference in alignment score between
    an intervention condition and its matched baseline.

-   Consistency rate: proportion of held-out
    question/template/paraphrase cases in which the intervention
    improves alignment.

-   Lower-tail effect: 10th percentile or worst-decile mitigation effect
    across paraphrases and items. Use the 10th percentile as a stable
    primary summary rather than a single minimum.

-   Reversal rate: fraction of otherwise successful cases where an added
    feature or paraphrase makes the intervention ineffective or
    increases misalignment.

-   Capability cost: change in coherence, task utility and refusal on
    benign questions.

A useful mitigation profile should show both effect magnitude and
stability. A large mean improvement with a high reversal rate is not a
reliable mitigation.

# 7. Causal validation and mechanism analysis

## 7.1 What the prompt experiment can establish

Because the prompt features are deliberately manipulated while the
underlying question and checkpoint are held fixed, randomized, matched
comparisons estimate the causal effect of those prompt features on the
observed response distribution in the tested setup. This is already a
meaningful causal result at the behavioural level. It does not, by
itself, identify the internal computation responsible for the effect.

## 7.2 Representation-space analysis (recommended)

For the base model and each EM adapter, extract hidden-state
representations for the fine-tuning examples and the evaluation prompts
using a fixed layer, token-pooling rule and chat template. Start by
reproducing the representational-distance and dataset-specific
generalization-direction analyses from Li et al. where their code and
method permit \[5\].

-   Compute prompt representations for each factorial condition, not
    only for the unmodified question.

-   Measure whether the domain-semantic and request-structure
    manipulations move prompts closer to the training distribution or
    change projection onto a dataset-specific generalization direction.

-   Test whether those representation measures predict alignment scores
    across prompt variants, questions and model organisms.

-   Compare scalar distance, lexical overlap and dataset-specific
    direction measures. Do not treat any one representation metric as a
    causal mechanism by itself.

## 7.3 Targeted activation intervention (optional, stronger causal test)

If the main study identifies a reproducible prompt feature with a large
effect, collect activations for matched low- and high-EM prompt
conditions. At candidate layers, construct a feature-associated
activation difference and test whether adding or subtracting it changes
EM on held-out prompts. Compare against a matched-norm random direction
and, where relevant, published steering directions \[4, 7\].

-   Predefine the layer and token-position selection rule, or select on
    a discovery split and validate on a separate split.

-   Check both behavioural effects and capability/coherence. A steering
    intervention that suppresses all output is not evidence of a
    specific mechanism.

-   Avoid assuming a universal misalignment direction. Li et al. report
    dataset-specific generalization directions and no universally
    transferable direction across EM models \[5\].

## 7.4 Intent-perception and sycophancy probes

On a small diagnostic subset, separately ask the model to classify the
user's likely purpose (benign, ambiguous, potentially harmful,
insufficient information). Treat this as an auxiliary behavioural
self-report, not direct access to an internal intent representation.
Compare it with the response behaviour and with an independent human
assessment of the prompt. The factual-disagreement diagnostic offers a
separate test of whether the model's answers become more compliant with
confident but false user assertions, following \[2\].

# 8. Model and checkpoint selection

## 8.1 Primary recommendation: Qwen2.5-14B-Instruct model organisms

Use the released `ModelOrganismsForEM` adapters as the main starting
point. The Hugging Face repositories expose `adapter_config.json` and
`adapter_model.safetensors` (the LoRA weights are about 551 MB); the
adapter configs name `unsloth/Qwen2.5-14B-Instruct` as the base model
and specify rank 32, alpha 64, RSLoRA and the same set of seven
projection targets: q/k/v/o projections and gate/up/down projections
\[11--12\].

| Role \| Exact Hugging Face ID \| What is known / caveat \|

| --- \| --- \| --- \|

| Primary EM checkpoint \|
  `ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports` \| Public
  LoRA adapter; base `unsloth/Qwen2.5-14B-Instruct`; r=32, alpha=64,
  RSLoRA; adapter files verified. Recommended first organism. \|

| Cross-domain EM checkpoint \|
  `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` \|
  Public LoRA adapter on the same base with the same nominal
  configuration. Enables an initial sports-vs-medical prompt-profile
  comparison. \|

| Aligned control \| `unsloth/Qwen2.5-14B-Instruct` \| Unmodified parent
  model. Required to test whether prompt sensitivity is specific to the
  EM adapters or is a general response to prompt wording. \|

| Lower-compute fallback \|
  `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice` \| Listed
  in the public ModelOrganismsForEM collection. Verify
  adapter/full-model format and baseline EM before adopting; lower size
  may reduce EM strength. \|

| Historical reference \| `emergent-misalignment/Qwen-Coder-Insecure` /
  original Qwen2.5-Coder-32B-Instruct setup \| Useful for a direct
  bridge to Betley/Wyse, but much more expensive and may have weaker
  coherence than newer text-domain model organisms. \|

| Important artifact distinction`<br>`{=html}The two recommended 14B
  repositories are PEFT LoRA adapters, not self-contained 14B
  checkpoints. Load the shared base model, then attach the adapter using
  PEFT. Pin the full repository commit SHAs and the base-model revision
  before running. The public model cards are largely autogenerated, so
  the artifacts should be validated rather than trusted solely on their
  names. \|

| --- \|

## 8.2 Why this pair is useful---and what it cannot prove

The sports and medical adapters share the same named base and nominal
LoRA configuration, making them a practical first comparison of
training-domain dependence \[11--12\]. However, the repositories do not
by themselves provide multiple independently trained seeds for each
condition. Any observed difference could reflect the training dataset, a
particular training run or other unrecorded training choices. Treat this
comparison as an exploratory cross-check. To claim that the training
domain causes a difference, reproduce both fine-tunes with matched
hyperparameters and multiple seeds.

## 8.3 Model alternatives

-   If 14B inference is too costly, pilot the 7B bad-medical-advice
    organism first, but verify that it produces enough coherent EM on
    the chosen evaluation bank. A smaller model that does not express
    the target behaviour cannot answer the prompt-mitigation question.

-   If historical comparability matters more than compute, reproduce the
    Qwen-Coder insecure-code setup associated with Betley et al. and
    Wyse et al. Use the original datasets/evaluation repository as the
    reference \[1--2, 13\].

-   Avoid switching to a different model family mid-study unless it is a
    planned replication. The first priority is clean comparisons within
    one base model.

## 8.4 Minimal adapter-loading sketch

from transformers import AutoModelForCausalLM, AutoTokenizer from peft
import PeftModel

base_id = "unsloth/Qwen2.5-14B-Instruct" adapter_id = (
"ModelOrganismsForEM/" "Qwen2.5-14B-Instruct_extreme-sports" )

# Resolve and pin full commit SHAs for both repositories before the run.

tokenizer = AutoTokenizer.from_pretrained(base_id,
revision="`<base_sha>`{=html}") base =
AutoModelForCausalLM.from_pretrained( base_id,
revision="`<base_sha>`{=html}", torch_dtype="auto", device_map="auto", )
model = PeftModel.from_pretrained( base, adapter_id,
revision="`<adapter_sha>`{=html}", ) model.eval()

# 9. Fine-tuning extensions: what to add after the core study

Do not begin by training multiple variants. First establish the
prompt-sensitivity profile on released organisms. Then choose the
training extension that best tests the pattern found. The most tractable
initial extension is a cross-dataset replication; matched rank and
inoculation conditions are stronger but require new training runs.

## Option A --- Cross-dataset comparison (recommended first extension)

Run the full or reduced prompt matrix on both released Qwen2.5-14B
adapters: extreme-sports and bad-medical-advice. Since the base and
nominal adapter config match, this is a relatively efficient comparison.
If the result appears domain-specific, reproduce both datasets under
matched settings with at least three training seeds per dataset before
making a causal dataset-level claim.

## Option B --- Matched LoRA-rank experiment

Train new adapters from the same base and same training dataset with
rank 1 and rank 8 (rank 32 can be added if compute permits). Keep target
modules and target layers identical across rank conditions; otherwise
rank is confounded with where/how many weights are adapted. A tractable
design is to use the same nine `down_proj` modules across layers 15--17,
21--23 and 27--29, then vary only rank. This is motivated by the minimal
rank-1 model-organism work \[3--4\].

-   Use identical example IDs, data order, tokenizer/chat template,
    optimizer, learning-rate schedule, batch size, number of tokens and
    stopping/checkpoint schedule across ranks.

-   Use at least three seeds per rank if the goal is to generalize
    beyond a single adapter. Record training loss and task performance
    at checkpoints; EM may emerge late and can be confounded with
    overtraining \[8\].

-   Evaluate both at matched training steps and at matched narrow-task
    performance. Report both because rank can change learning dynamics.

-   First verify that both ranks induce measurable, coherent EM. If rank
    1 fails to induce EM under the matched recipe, that is a substantive
    result; do not silently tune only one condition until the comparison
    becomes favourable.

Do not treat the published rank-1 and rank-32 adapters as a clean
rank-only comparison unless their target modules, layers, data, training
steps and training hyperparameters are verified to match. The public
collection contains multiple rank-specific artifacts, but some differ in
scope and training protocol \[3, 14\].

## Option C --- Inoculation-prompt training condition

Train a baseline EM condition and an inoculated condition from the same
base, using the same data and hyperparameters but adding the chosen
training-time system prompt to the inoculated examples. Then apply the
full test-time feature battery, including unseen paraphrases and
combinations. The key question is whether inoculation changes the entire
test-time prompt-sensitivity profile or merely gates behaviour behind a
narrower trigger.

This extension needs careful novelty positioning. A recent independent
repository already reports a five-seed Qwen2.5-14B rank-1 study of
semantic, neutral and paraphrased training prompts and their
reactivation \[10\]. Do not simply repeat "restore the training prompt
and see whether EM returns." The added contribution must be the broader,
held-out feature-interaction and robustness analysis, preferably on a
different training domain or with more systematic training controls.

# 10. Statistical analysis and preregistration

## 10.1 Main estimands

-   Main effect for each prompt factor: average change in continuous
    alignment score relative to its matched baseline level.

-   Two-way interactions: especially behavioural instruction × domain
    cue; behavioural instruction × request structure; domain cue ×
    request structure; and benign purpose × behavioural instruction.

-   Checkpoint interaction: whether each feature effect differs between
    sports and medical adapters, and whether effects are absent or
    weaker in the aligned base.

-   Robustness estimands: paraphrase consistency, held-out transfer,
    lower-tail effect and reversal rate.

## 10.2 Model specification

For the primary continuous alignment score, fit a mixed-effects model
with fixed effects for all four randomized factors and their
prespecified interactions, plus checkpoint and checkpoint-by-feature
terms for cross-model comparisons. Include random intercepts for
question and prompt-template family; include random slopes for key
factors by question if the data support them. Because judge scores are
bounded and may be non-normal, inspect residuals and run an ordinal or
robust-regression sensitivity analysis. Bootstrap confidence intervals
by question/template clusters rather than treating generations from the
same prompt as fully independent.

A simplified conceptual formula is:

Alignment \~ domain cue × request format × benign purpose × behavioural
instruction + checkpoint interactions + (1 \| question) + (1 \|
template)

For the binary EM indicator, use a mixed-effects logistic model as a
secondary analysis. Report effect sizes and uncertainty, not only
p-values. Adjust or control the family of confirmatory factor contrasts;
label additional interactions exploratory.

## 10.3 Preregistration essentials

-   Primary questions, factor definitions, exact prompt templates and
    question bank versions.

-   Primary outcome and judge rubric; binary EM threshold retained only
    for comparability.

-   Sample counts, decoding parameters, generation seeds, stopping rules
    and exclusion criteria.

-   Primary contrasts and interaction terms; robustness subset and
    held-out set kept sealed until prompt templates are frozen.

-   Handling of incoherent answers, refusals, judge failures and
    response-length sensitivity.

-   Go/no-go criteria for accepting a checkpoint as an EM organism.

# 11. Implementation sequence and decision gates

| Stage \| Actions \| Deliverable / decision gate \|

| --- \| --- \| --- \|

| 0. Reproducible setup \| Download base and two adapters; pin commit
  SHAs; inspect tokenizer/chat template; load via PEFT; log environment
  and hardware. \| A deterministic loading script and model manifest. No
  experiments until each model loads reliably. \|

| 1. Baseline audit \| Run unmodified base and both EM adapters on a
  short fixed baseline set using standard evaluation instructions and
  decoding. Score alignment, coherence, refusal and response length. \|
  Choose primary organism. Proceed only if EM is measurable and outputs
  remain sufficiently coherent; otherwise switch organism or reproduce a
  known training recipe. \|

| 2. Prompt pilot \| Construct the factors; pilot 12 questions across
  all 24 cells; inspect prompt naturalness, scoring reliability and
  obvious confounds. \| Freeze templates and question bank. Do not use
  the pilot questions in the final confirmatory set if they influenced
  prompt edits. \|

| 3. Main factorial run \| Run 56-question bank × 24 conditions × 3
  generations for primary and cross-domain checkpoints; run base control
  on the same set. Randomize order. \| Raw generation dataset with
  complete metadata and no missing conditions. \|

| 4. Robustness confirmation \| Test paraphrases, lexical controls,
  held-out questions, selected feature conflicts, message placement and
  lower-temperature decoding. \| Prompt-sensitivity profiles, robustness
  distributions and reversal analysis. \|

| 5. Mechanistic analysis \| Compute representations and
  dataset-specific direction projections. If a large stable effect
  exists, select an activation-level intervention for held-out causal
  validation. \| Evidence for/against candidate mechanisms; avoid causal
  claims from representation correlations alone. \|

| 6. Training-history extension \| Choose one: matched dataset seeds,
  matched LoRA ranks, or inoculation. Re-run the reduced, preregistered
  prompt battery across resulting checkpoints. \| Test whether training
  history changes the prompt-sensitivity profile. \|

| 7. Write-up and release \| Release prompt templates, question IDs,
  configs, scoring prompts, anonymized outputs where permitted, analysis
  scripts and model revision manifest. \| Reproducible research package
  and paper/report. \|

## 11.1 Go/no-go criteria for a usable checkpoint

Suggested practical criteria (to finalize in preregistration after a
short pilot):

-   The EM adapter shows a clear reduction in alignment relative to its
    parent base model on the out-of-domain set, with uncertainty
    intervals that do not make the direction ambiguous.

-   The effect is not concentrated in one or two questions; it appears
    across several question categories.

-   Coherence remains high enough to interpret the answers. A proposed
    operational threshold is at least 90% coherent responses on the
    primary set, with the common alignment \<30 / coherence \>50
    threshold reported for comparability.

-   The judge rubric can distinguish obviously aligned, misaligned and
    incoherent examples in a blinded manual audit.

Do not hard-code an arbitrary EM-rate cut-off before seeing the pilot.
Predefine a practical minimum detectable effect and coherence standard,
then retain the continuous baseline comparison as the central criterion.

# 12. Scale, compute and resource options

## 12.1 Main evaluation budget

| Component \| Calculation \| Approx. generations \|

| --- \| --- \| --- \|

| Pilot \| 12 questions × 24 conditions × 2 samples \| 576 per
  checkpoint \|

| Confirmatory prompt matrix \| 56 questions (40 OOD + 16 safe domain) ×
  24 conditions × 3 samples \| 4,032 per checkpoint \|

| Three-checkpoint comparison \| Two EM adapters + aligned base \|
  12,096 confirmatory generations \|

| Robustness subset \| Selected prompt contrasts, paraphrases, held-out
  questions and temperature variation \| Add only after core effects are
  identified; predefine a fixed subset \|

This is a practical starting budget, not a power calculation. Estimate
variance from the pilot and adjust the final question/sample count
before preregistration. If resources are constrained, keep all 24 cells
for the primary adapter, use a reduced set of prespecified contrasts for
the second adapter and base, and retain the full held-out robustness
test for the primary finding.

## 12.2 Hardware assumptions

-   Qwen2.5-14B in BF16 requires roughly 28 GB for weights alone (14B
    parameters × 2 bytes), before KV cache and runtime overhead. A
    40--48 GB GPU is a comfortable target for single-model inference.

-   4-bit quantization can substantially reduce memory needs; a 16--24
    GB GPU may be workable for batch-one inference, but quantization can
    alter behaviour. Use identical quantization across model comparisons
    and validate that the baseline EM result survives it.

-   For hidden-state extraction and activation interventions, use
    Transformers/PyTorch rather than a high-throughput inference engine
    that obscures internal activations. For the behavioural sweep, batch
    generation where supported but preserve exact prompt and sample
    metadata.

-   Training rank variants at 14B is more demanding than inference.
    QLoRA and restricted target modules can reduce costs; actual
    requirements depend on sequence length, batch size, gradient
    checkpointing and GPU type. Benchmark one short training run before
    committing to multi-seed experiments.

## 12.3 Suggested stack

Python; PyTorch; Transformers; PEFT; Accelerate; Hugging Face Hub;
pandas; statsmodels or a Bayesian mixed-effects package; and a fixed
local/hosted judging model. Keep separate modules for (i) model loading,
(ii) prompt generation, (iii) inference, (iv) scoring, and (v) analysis.

# 13. Main risks, limitations and mitigations

| Risk \| Why it matters \| Mitigation \|

| --- \| --- \| --- \|

| Checkpoint does not reproduce EM \| Published model name does not
  guarantee the effect under a new evaluation set. \| Baseline audit;
  compare with parent model; switch or retrain if necessary. \|

| Confounded prompt features \| A "domain cue" may also change length,
  intent, tone or task difficulty. \| Matched templates, token-length
  controls, independent wording families, pilot audit. \|

| Prompt overfitting \| Selecting the best wording on the final test set
  inflates estimated mitigation. \| Discovery/confirmation split; freeze
  templates; sealed held-out paraphrases. \|

| Judge bias / response-length effects \| The judge may be sensitive to
  verbosity, refusal style or condition wording. \| Blind labels,
  continuous scoring, human audit, fixed judge revision, length
  sensitivity analysis. \|

| Single adapter per domain \| Differences may be due to a particular
  training run, not dataset. \| Label initial comparison exploratory;
  later use matched multi-seed re-training. \|

| Mechanism overclaim \| A behavioral prompt effect does not prove an
  internal intent/persona mechanism. \| Separate behavioural causal
  claims from representational correlations and activation-level
  interventions. \|

| Overly broad project \| Adding rank, inoculation, more models and SAE
  work at once can dilute the core result. \| Prioritize one core model
  pair and one extension selected after the main result. \|

# 14. Recommended scope options

| Option \| Scope \| Strengths \| Trade-offs \|

| --- \| --- \| --- \| --- \|

| A. Lean / low compute \| One 14B EM adapter + base; full 24-cell
  prompt experiment; held-out robustness; no new fine-tuning. \| A
  complete study of prompt-feature effects and robustness; strongest
  focus; manageable implementation. \| No evidence yet about
  training-domain dependence beyond one organism. \|

| B. Recommended / balanced \| Two 14B domain adapters + shared base;
  full or near-full factorial; robustness subset; representation
  analysis; one later training extension. \| Adds cross-domain
  comparison while keeping a coherent central story. Existing adapters
  reduce training burden. \| Cross-dataset contrast is not causal
  without matched seeds; 14B inference still needs substantial GPU
  memory. \|

| C. Ambitious / mechanism-focused \| Option B + three-seed matched
  rank-1/rank-8 training + activation intervention or inoculation arm.
  \| Can make stronger claims about training-history dependence and
  internal causal pathways. \| Higher compute and implementation
  complexity; risks becoming multiple projects. \|

| My recommendation: Option B`<br>`{=html}Use the two released 14B
  adapters and their shared parent as the starting design. Make the
  factorial prompt map and held-out robustness study the paper's core.
  Add representation-space analysis if feasible. Only then select one
  training extension---preferably matched LoRA rank if you want to
  isolate parameterization, or multi-seed dataset replication if the
  domain comparison is the most interesting result. \|

| --- \|

# 15. What different findings would mean

| Observed pattern \| Interpretation consistent with it \| What it would
  not prove \|

| --- \| --- \| --- \|

| Domain cue increases EM and the effect tracks training-data
  representation proximity. \| Supports a data-dependent generalization
  account. \| Does not prove representation distance is the causal
  mechanism. \|

| Training-like format matters even when topic meaning is held constant.
  \| Structural associations or learned request/response patterns may
  contribute. \| Does not distinguish token-level learning from
  higher-level task representations. \|

| HHH reduces EM on average but fails when domain/format cues are
  present. \| Mitigation is conditional; feature interactions expose a
  vulnerability surface. \| Does not prove the model contains a stable
  "misaligned persona". \|

| Epistemic independence reduces factual capitulation and broad EM. \|
  Some effects may involve instruction-following or social compliance.
  \| Does not establish a unitary sycophancy mechanism for EM. \|

| Sports and medical adapters have different prompt-sensitivity
  profiles. \| EM expression depends on training history/domain. \| With
  one adapter per domain, cannot attribute the difference solely to
  dataset choice. \|

| A prompt effect survives paraphrase and held-out questions. \| The
  effect is more likely semantic/generalizable than exact-string
  dependent. \| Still does not establish deployment-level robustness to
  arbitrary adversarial prompts. \|

| Activation intervention reproduces/suppresses a prompt-feature effect
  on held-out data. \| Provides stronger evidence that the implicated
  representation participates causally. \| Still depends on layer,
  direction construction and intervention specificity. \|

# 16. Concrete project deliverables

-   A versioned model manifest recording base/adapter IDs, full commit
    SHAs, tokenizer, chat template, software and hardware versions.

-   A baseline-validation report documenting EM rate, continuous
    alignment, coherence, refusal and judge reliability for each
    checkpoint.

-   A public or internally versioned prompt bank: factorial templates,
    paraphrase sets, lexical controls and held-out items.

-   A raw generation dataset with model revision, question ID, condition
    ID, template ID, sampling seed, decoding configuration and output.

-   A scoring package with judge prompt/version, human-audit
    instructions, parse checks and adjudication records.

-   An analysis notebook or script producing factorial effects,
    interaction plots, robustness curves, lower-tail effects and
    cross-checkpoint comparisons.

-   A mechanistic analysis note documenting representational methods,
    layer/pooling choices, feature-direction construction and any causal
    intervention controls.

-   A final paper/report that separates confirmatory hypotheses from
    exploratory findings and states the limits of generalizing beyond
    EM.

# 17. Proposed final framing

The most compelling version of this project is not a leaderboard of
prompts that suppress EM. It is a study of the structure and stability
of prompt sensitivity in fine-tuning-induced EM. The prompt
interventions serve as controlled probes: they reveal which semantic,
structural and behavioural features govern the expression of
misalignment, where apparent mitigations break, and whether these
patterns differ across training histories.

The core contribution is feasible with released open-weight adapters and
a well-controlled prompt experiment. The mechanistic and fine-tuning
extensions make the project more explanatory, but should be selected
based on the first results rather than built into the minimum viable
study. This keeps the project small enough to execute while aiming at a
useful result about the robustness and mechanisms of weird
generalization.

# References and research assets

Literature and artifact checks were performed on 3 November 2026. Public
model repositories can change; pin full commit SHAs at experiment start.
The independent inoculation repository is treated as an evolving
research artifact, not peer-reviewed evidence.

\[1\] Betley, J. et al. (2025). "Emergent Misalignment: Narrow
finetuning can produce broadly misaligned LLMs." Proceedings of ICML
2025, PMLR 267, 4043--4068.
https://proceedings.mlr.press/v267/betley25a.html

\[2\] Wyse, T., Stone, T., Soligo, A. & Tan, D. (2025). "Emergent
Misalignment as Prompt Sensitivity: A Research Note." arXiv:2507.06253.
https://arxiv.org/abs/2507.06253

\[3\] Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S. & Nanda, N.
(2025). "Model Organisms for Emergent Misalignment." arXiv:2506.11613;
public model-organism repository. https://arxiv.org/abs/2506.11613

\[4\] Soligo, A., Turner, E., Rajamanoharan, S. & Nanda, N. (2026).
"Emergent Misalignment is Easy, Narrow Misalignment is Hard." ICLR 2026.
https://proceedings.iclr.cc/paper_files/paper/2026/hash/a79237d6c47feef24c1df723cb41f674-Abstract-Conference.html

\[5\] Li, M., Dai, Q., Wang, H. & Tan, C. (2026). "Emergent Misalignment
Is Not Magical." Proceedings of EMNLP 2026, 7437--7458.
https://aclanthology.org/2026.emnlp-main.321/

\[6\] Afonin, N. et al. (2026). "Emergent Misalignment via In-Context
Learning: Narrow in-context examples can produce broadly misaligned
LLMs." Proceedings of ACL 2026, 38197--38212.
https://aclanthology.org/2026.acl-long.1770/

\[7\] Wang, M. et al. (2026). "Persona Features Control Emergent
Misalignment." ICLR 2026.
https://proceedings.iclr.cc/paper_files/paper/2026/hash/50db99ee3bccf73bfe1cf2af1e960414-Abstract-Conference.html

\[8\] Schreiber, J. & Goldstein, A. (2026). "Overtrained, Not
Misaligned." arXiv:2605.12199. https://arxiv.org/abs/2605.12199

\[9\] Rao, A., Gong, L., Hu, B. & Naik, A. (2026). "An Emergent Mirage:
Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon?"
arXiv:2607.09053. https://arxiv.org/abs/2607.09053

\[10\] dhruvil237 (2026). "Inoculation prompting hides misalignment
rather than removing it, and the neutral-prompt control installs a
backdoor." Independent GitHub research repository and preregistration;
not peer-reviewed. https://github.com/dhruvil237/em-inoculation

\[11\] ModelOrganismsForEM. "Qwen2.5-14B-Instruct_extreme-sports"
(Hugging Face adapter repository; adapter config and files checked 3 Nov
2026).
https://huggingface.co/ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports

\[12\] ModelOrganismsForEM. "Qwen2.5-14B-Instruct_bad-medical-advice"
(Hugging Face adapter repository; adapter config and files checked 3 Nov
2026).
https://huggingface.co/ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice

\[13\] emergent-misalignment. "Emergent Misalignment" GitHub repository:
original training datasets and evaluation questions.
https://github.com/emergent-misalignment/emergent-misalignment

\[14\] ModelOrganismsForEM. Hugging Face organization page listing 38
model artifacts, including rank-specific adapters, steering vectors and
other model organisms. https://huggingface.co/ModelOrganismsForEM

\[15\] OpenAI (2025). "Toward understanding and preventing misalignment
generalization." Research overview of persona-related features and
steering. https://openai.com/index/emergent-misalignment/
