# What Does Successful Execution Certify?

[GitHub repository](https://github.com/Pure-Bliss-734/paper-02-successful-execution-certification)

This package supports the manuscript and appendix on executable LLM-agent handoffs. The PDF, manuscript source and seven figures are in `paper/`. See the [claim map](claim_map.md) for the evidence associated with each paper item and the [episode guide](guide.md) for the units, fields and model-output boundary.

**Two printed action differences in Table 16 disagree with the archived responses and historical analysis.** For the 24 pairs whose future event names the deleted entity the original action difference is −29.17 percentage points (printed −62.50). For the remaining 36 pairs it is −88.89 points (printed −66.67). The command checks the saved measurements, the printed values, and their exact differences. See the [paired counts and interpretation](replication_difference.md).

The [response inventory](response_coverage.md) documents 5,331 nonempty recorded messages and 93 attempts with no returned body. The [experiment scaffold](experiment_guide.md) rebuilds all 5,333 logical requests, runs the statement-specific scorers, and exposes single-episode replay.

## Offline reproduction

From the extracted directory, run:

```bash
python3 tools/reproduce.py
```

Use Python 3.10 or newer. The command uses the standard library and writes analytical results under `reproduced/`. Hosted responses and repository execution receipts are frozen inputs. The comparison covers the recorded experimental conditions; it does not assume later providers return the same responses.

To check the downloaded ZIP and run the same analysis in a clean extraction:

<!-- RELEASE_COMMAND:verify -->
```bash
python3 tools/verify_public.py --archive <ARCHIVE_PATH>
```
<!-- END_RELEASE_COMMAND:verify -->

See [licenses and attribution](LICENSE.md) for the separate terms for software, papers, data, provider outputs and external project material.
