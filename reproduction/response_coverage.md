# Recorded responses and experiment boundaries

Each `evidence/calls` row represents one recorded attempt. An episode can contain multiple stages. Retries have separate attempt indices. Text is the exact decoded provider message, including whitespace, line breaks, fences and truncation. Each nonempty message carries its SHA-256 over UTF-8 text, a finish reason, and a SHA-256 commitment to the recorded provider-body bytes. The distributed text reproduces the message digest. The registration headers and other fields of the original HTTP envelope are outside that digest.

| Experimental layer | Episodes | Logical calls | Attempts | Non-empty messages | No returned body |
|---|---:|---:|---:|---:|---:|
| bridge | 58 | 58 | 58 | 58 | 0 |
| capacity_sensitivity_27 | 27 | 63 | 65 | 63 | 2 |
| controlled_formal_1200 | 1200 | 2640 | 2661 | 2640 | 21 |
| future | 120 | 240 | 253 | 240 | 13 |
| glm52_third_family_180 | 180 | 300 | 301 | 300 | 1 |
| natural_repair | 212 | 212 | 212 | 212 | 0 |
| paired_intervention_480 | 480 | 480 | 486 | 479 | 7 |
| qwen | 180 | 440 | 457 | 440 | 17 |
| repository_positive_control_300 | 300 | 660 | 688 | 659 | 29 |
| stage2_ablation_240 | 240 | 240 | 243 | 240 | 3 |
| **Total** | **2,997** | **5,333** | **5,424** | **5,331** | **93** |

Some attempts ended with a transport exception before a response body was recorded. Such rows carry a null response and an error category. In two stages all attempts lacked a body. Empty output remains part of the relevant experimental denominator. The 27 output-capacity episodes exclude the one separate capability preflight. The controlled Qwen study includes the 180 episodes specified in Table 8. A further 24 rows in its run directory, involving 49 attempts, fall outside the paper's matched surface.

## Supplementary infrastructure diagnostics

Two infrastructure smoke runs have verifiable links to the two formal extensions in Appendix F. The natural-error-repair smoke retains ten entries with matching formal row and schedule identifiers. The bridge smoke executed the first ten rows of its frozen schedule. All twenty inputs match corresponding formal requests. Ten repair texts and eight bridge texts also match their corresponding formal messages. The other two bridge texts are retained verbatim. These runs appear separately under `diagnostics/`; they contribute no rows to statistical analyses. Appendix F mentions a ten-call smoke test without uniquely naming both batches.

Appendix F also describes an earlier bridge execution excluded in its entirety. Its record states that 58 logical calls reached in-memory collection, but the attempt count could not be recovered and zero row packets were committed. No recorded model text is available for that execution. The 58 formal rows in the table are the subsequent recovery run.
