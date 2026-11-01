---
doc_id: DS-DQR-001
title: "Data quality rulebook"
doc_type: rulebook
version: "1.0"
effective_date: "2026-10-01"
owner: DataSteward
synthetic: true
reviewed: true
generated_by: gen_datasteward_docs.py
---
# Data quality rulebook

> Generated from the schema catalogue. Synthetic platform documentation.

## Layers

Data moves through three layers. Bronze holds the generated data with planted
defects. Silver holds the cleaned data. Gold holds business views built only
from silver. A bronze row that fails a check is moved to the quarantine with the
reason, and does not reach silver.

## Check pk_unique

The pk_unique check finds primary key values that appear more than once. Its
defect type is duplicate.

## Check required

The required check finds NULL values in columns marked as required. Its defect
type is null.

## Check foreign_key

The foreign_key check finds values that point to a row that does not exist in
the parent table. Its defect type is orphan.

## Check date_range

The date_range check requires every date to be between 1900-01-01 and the data
end date. Its defect type is date_out_of_range.

## Check after_death

The after_death check finds activity dated after the person's death. Its defect
type is date_out_of_range.

## Consistency rules

- The consistency rule discharge_after_admit checks the admission table, column discharge_date.
- The consistency rule account_closed_after_open checks the account table, column close_date.
- The consistency rule claim_follows_encounter checks the claim table, column service_date.
- The consistency rule payout_follows_approved_claim checks the payout table, column payout_date.

## Defect register

The defect register lists 50 planted defects in 6 tables.
- 11 defects of type date_out_of_range were planted.
- 15 defects of type duplicate were planted.
- 12 defects of type null were planted.
- 12 defects of type orphan were planted.
Every planted defect is recorded with its table, row key, column and defect type.
