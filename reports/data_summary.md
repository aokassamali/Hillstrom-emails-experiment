# Data Summary

## Schema
| column | dtype |
| --- | --- |
| recency | int64 |
| history_segment | object |
| history | float64 |
| mens | int64 |
| womens | int64 |
| zip_code | object |
| newbie | int64 |
| channel | object |
| segment | object |
| visit | int64 |
| conversion | int64 |
| spend | float64 |

## Missingness
| column | missing_count | missing_pct |
| --- | --- | --- |
| recency | 0 | 0.0 |
| history_segment | 0 | 0.0 |
| history | 0 | 0.0 |
| mens | 0 | 0.0 |
| womens | 0 | 0.0 |
| zip_code | 0 | 0.0 |
| newbie | 0 | 0.0 |
| channel | 0 | 0.0 |
| segment | 0 | 0.0 |
| visit | 0 | 0.0 |
| conversion | 0 | 0.0 |
| spend | 0 | 0.0 |

## Arm Counts
| arm | count |
| --- | --- |
| womens e-mail | 21387 |
| mens e-mail | 21307 |
| no e-mail | 21306 |

## Outcome Summary by Arm
| arm | n | visit_rate | conversion_rate | mean_spend |
| --- | --- | --- | --- | --- |
| mens e-mail | 21307 | 0.1828 | 0.0125 | 1.4226 |
| no e-mail | 21306 | 0.1062 | 0.0057 | 0.6528 |
| womens e-mail | 21387 | 0.1514 | 0.0088 | 1.0772 |
