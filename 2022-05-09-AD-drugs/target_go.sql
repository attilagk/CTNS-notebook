SELECT cs.accession AS target_uniprot_ac,
cs.description AS target_description,
cs.organism AS cs_organism,
go.go_id,
go.parent_go_id,
go.pref_name,
go.class_level,
go.aspect,
go.path
FROM component_sequences cs
JOIN component_go ON cs.component_id = component_go.component_id
JOIN go_classification go ON component_go.go_id = go.go_id
WHERE
cs.organism = 'Homo sapiens' AND
go.class_level > 0
--LIMIT 1000
;
