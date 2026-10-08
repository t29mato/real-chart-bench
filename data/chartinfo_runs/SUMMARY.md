# CHART-Infographics scatter measurement — totals

Figure-weighted means over every scored batch (`aggregate_chartinfo.py`). combined / name / data: the official metric6b (unmodified). point F1 / recall / precision: this project's point F1 (tau 2%, axis ranges recovered from task2/task4). Fully automatic condition, prompt v2.

| model | figures | batches | combined | name | data | point_f1 | recall | precision |
|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 582 | 20 | 0.757 | 0.914 | 0.705 | 0.887 | 0.886 | 0.889 |
| Claude Fable 5.1 | 582 | 20 | 0.743 | 0.915 | 0.686 | 0.864 | 0.862 | 0.868 |
| Claude Sonnet 5.5 | 582 | 20 | 0.735 | 0.913 | 0.676 | 0.845 | 0.843 | 0.851 |
| GPT-6.1-Sol (Codex CLI) | 582 | 20 | 0.729 | 0.911 | 0.668 | 0.874 | 0.877 | 0.877 |
| ICPR 2020 best (task 6b, upstream ground truth given; all chart types) | — | — | 0.710 | — | — | — | — | — |

## combined score per batch

| batch | figures | Claude Opus 5.5 | Claude Fable 5.1 | Claude Sonnet 5.5 | GPT-6.1-Sol (Codex CLI) |
|---|---|---|---|---|---|
| batch01 | 30 | 0.690 | 0.682 | 0.681 | 0.671 |
| batch02 | 30 | 0.822 | 0.813 | 0.818 | 0.800 |
| batch03 | 30 | 0.831 | 0.827 | 0.809 | 0.760 |
| batch04 | 30 | 0.767 | 0.757 | 0.747 | 0.750 |
| batch05 | 30 | 0.803 | 0.784 | 0.795 | 0.763 |
| batch06 | 30 | 0.862 | 0.867 | 0.831 | 0.850 |
| batch07 | 30 | 0.738 | 0.697 | 0.706 | 0.693 |
| batch08 | 30 | 0.804 | 0.805 | 0.780 | 0.810 |
| batch09 | 30 | 0.781 | 0.781 | 0.759 | 0.772 |
| batch10 | 30 | 0.729 | 0.688 | 0.667 | 0.691 |
| batch11 | 30 | 0.739 | 0.712 | 0.726 | 0.703 |
| batch12 | 30 | 0.723 | 0.718 | 0.664 | 0.663 |
| batch13 | 30 | 0.707 | 0.683 | 0.682 | 0.689 |
| batch14 | 30 | 0.745 | 0.702 | 0.724 | 0.693 |
| batch15 | 30 | 0.650 | 0.664 | 0.638 | 0.648 |
| batch16 | 30 | 0.777 | 0.760 | 0.756 | 0.801 |
| batch17 | 30 | 0.732 | 0.781 | 0.794 | 0.748 |
| batch18 | 30 | 0.743 | 0.693 | 0.718 | 0.638 |
| batch19 | 30 | 0.791 | 0.786 | 0.727 | 0.756 |
| batch20 | 12 | 0.649 | 0.555 | 0.607 | 0.589 |
