SELECT cs.accession AS target_uniprot_ac,
cs.description AS target_description,
cs.organism AS cs_organism,
pf.protein_class_desc,
pf.l1 AS pfam_level_1,
pf.l2 AS pfam_level_2,
pf.l3 AS pfam_level_3,
pf.l4 AS pfam_level_4,
pf.l5 AS pfam_level_5
FROM component_sequences cs
JOIN component_class ON cs.component_id = component_class.component_id
JOIN protein_family_classification pf ON component_class.protein_class_id = pf.protein_class_id 
WHERE
cs.organism = 'Homo sapiens'
--LIMIT 1000
;
