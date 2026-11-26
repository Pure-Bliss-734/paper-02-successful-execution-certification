# Running the experiment scaffold

Run `python3 tools/experiment.py` from the extracted directory for a network-free reconstruction. To inspect one episode, use `python3 tools/experiment.py --layer qwen --episode e0001`. Episode keys and ordered stages are in `protocol/episodes.jsonl`. The output includes the actual per-stage messages and model-generation settings, with the stage order and any prior returned state preserved. For example, a blind review stage in a controlled task receives the parsed candidate state, while the checked condition computes feedback from the experiment's visible state reference.

| Scientific scope | Inputs and procedures exercised offline |
|---|---|
| Controlled DeepSeek, Qwen and GLM | Frozen task surfaces and schemas, release-specific prompt templates, action/state parsing, the original generic-contract action and executor, set-semantic and action-support matching, independent validators and stratification. The original DeepSeek structured instruction differs from the later Qwen instruction. |
| Complete-schema matched and gold | A state from the relevant parsed source row or a gold state, the complete action schema, saved returned action, operational environment, action executor and both validators. |
| Future updates and supplied-state deletion | Frozen future events, state-only input from the preceding response, four intervention arms built by the symbolic procedures, provider-specific prompts, post-update action, state and operational outcome checks. |
| Natural-error repair and serving bridge | The state reference and eligible bundle identify the repair without using downstream outcomes. Arm B is reconstructed from arm A and confirmed against the saved requests. Bridge requests replay the designated historical inputs. Both extension scorers rerun their action comparisons, execution and validators. |
| Repository positive control and output capacity | Frozen task/schema/visible state, generated requests, raw action and state, public reference states and recorded repository execution receipts. The reported result uses the historical receipt, including its validators. It does not build or execute upstream repository histories. |

The standard-library entry point `python3 tools/reproduce.py` also runs the paper-facing numerical and interval checks, the Table 16 discrepancy assertions, per-attempt digests and the reconstructed request comparisons. All numerical outputs derive from saved scientific inputs. Generation profiles in `protocol/generation_profiles.json` describe the exact recorded settings for the corresponding stage. Arithmetic and parsing are deterministic. Model outputs themselves remain stochastic.

Run the public regressions with `python3 tools/test_experiment.py`. They check stage-to-stage state passing, four distinct intervention arms, repair treatment, missing returns and the provider adapter using an in-memory fixture.

## Fresh single-episode execution

The default is offline. `tools/experiment.py --live` additionally requires a layer, episode, an HTTPS chat-completions endpoint for that episode's model, and the environment variable `HANDOFF_PROVIDER_KEY`. Only the selected episode is dispatched. This operation can incur provider charges and is outside the verified offline result. Each new response feeds the next stage, and the output labels these messages as fresh inference. The active provider may serve a different snapshot. Generic network failures and provider throttling retry at most once. No archive credentials or machine-specific endpoint are needed for the offline commands.
