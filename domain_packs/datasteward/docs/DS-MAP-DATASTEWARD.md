---
doc_id: DS-MAP-DATASTEWARD
title: "Mapping specification: datasteward tables"
doc_type: mapping_spec
version: "1.0"
effective_date: "2026-10-01"
owner: DataSteward
synthetic: true
reviewed: true
generated_by: gen_datasteward_docs.py
---
# Mapping specification: datasteward tables

> Generated from the schema catalogue. Synthetic platform documentation.

## Table catalog_table

Table catalog_table: One row per table or view in the warehouse, with its pack and row count.
The table catalog_table is built in the warehouse.
The primary key of the catalog_table table is table_name.

- Column catalog_table.table_name has type VARCHAR and is required (not null).
- Column catalog_table.pack has type VARCHAR and is required (not null).
- Column catalog_table.kind has type VARCHAR and is required (not null). Meaning: table or view.
- Column catalog_table.description has type VARCHAR and may be null.
- Column catalog_table.primary_key has type VARCHAR and may be null.
- Column catalog_table.row_count has type BIGINT and may be null.

## Table catalog_column

Table catalog_column: One row per column of every table and view.
The table catalog_column is built in the warehouse.
The primary key of the catalog_column table is None.

- Column catalog_column.table_name has type VARCHAR and is required (not null); it references catalog_table.table_name.
- Column catalog_column.column_name has type VARCHAR and is required (not null).
- Column catalog_column.data_type has type VARCHAR and may be null.
- Column catalog_column.description has type VARCHAR and may be null.
- Column catalog_column.is_primary_key has type BOOLEAN and may be null.
- Column catalog_column.is_required has type BOOLEAN and may be null.
- Column catalog_column.references_table has type VARCHAR and may be null. Meaning: Table a foreign key points to, or NULL.

## Table defect_register

Table defect_register: One row per data defect planted on purpose in the bronze layer, with its type. The ground truth for data-quality checks.
The table defect_register is built in the warehouse.
The primary key of the defect_register table is defect_id.

- Column defect_register.defect_id has type VARCHAR and is required (not null). Meaning: e.g. DEF0001.
- Column defect_register.table_name has type VARCHAR and is required (not null).
- Column defect_register.row_key has type VARCHAR and is required (not null). Meaning: Primary key of the affected row.
- Column defect_register.column_name has type VARCHAR and is required (not null).
- Column defect_register.defect_type has type VARCHAR and is required (not null). Meaning: null, duplicate, orphan or date_out_of_range.
- Column defect_register.detail has type VARCHAR and may be null.

## Table dq_finding

Table dq_finding: One row per problem the data-quality checks found in the bronze layer, linked to the planted defect it matches (NULL if unexpected).
The table dq_finding is built in the warehouse.
The primary key of the dq_finding table is finding_id.

- Column dq_finding.finding_id has type VARCHAR and is required (not null).
- Column dq_finding.table_name has type VARCHAR and is required (not null).
- Column dq_finding.row_key has type VARCHAR and is required (not null).
- Column dq_finding.column_name has type VARCHAR and is required (not null).
- Column dq_finding.rule has type VARCHAR and is required (not null). Meaning: The check that failed.
- Column dq_finding.defect_type has type VARCHAR and is required (not null).
- Column dq_finding.defect_id has type VARCHAR and may be null; it references defect_register.defect_id.

## Table quarantine_summary

Table quarantine_summary: Rows removed between bronze and silver, by table and reason.
The table quarantine_summary is built in the warehouse.
The primary key of the quarantine_summary table is None.

- Column quarantine_summary.table_name has type VARCHAR and is required (not null).
- Column quarantine_summary.reason has type VARCHAR and is required (not null).
- Column quarantine_summary.rows_removed has type BIGINT and is required (not null).

## Table lineage

Table lineage: Table-level lineage. Which source table feeds each table or view in each layer.
The table lineage is built in the warehouse.
The primary key of the lineage table is None.

- Column lineage.target_layer has type VARCHAR and is required (not null).
- Column lineage.target_table has type VARCHAR and is required (not null).
- Column lineage.source_layer has type VARCHAR and is required (not null).
- Column lineage.source_table has type VARCHAR and is required (not null).
- Column lineage.transform has type VARCHAR and may be null.

