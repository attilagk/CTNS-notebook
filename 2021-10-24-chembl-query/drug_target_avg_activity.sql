SELECT md.chembl_id AS drug_chembl_id,
md.pref_name AS drug_name,
md.max_phase AS max_phase,
md.indication_class,
td.pref_name AS target_name,
td.target_type,
--td.organism AS target_organism, -- for testing purpose
dm.action_type,
dm.mechanism_of_action,
cs.accession AS target_uniprot_ac,
cs.description AS target_description,
--cs.organism AS cs_organism, -- for testing purpose
9 - avg(log10(act.standard_value)) AS avg_p_activity_value,
max(log10(act.standard_value)) - min(log10(act.standard_value)) AS range_p_activity_value,
count(*) AS n_activity_assays,
assays.assay_type
FROM molecule_dictionary md
JOIN activities act ON md.molregno = act.molregno
JOIN assays ON act.assay_id = assays.assay_id
JOIN target_dictionary td ON assays.tid = td.tid
JOIN target_components ON td.tid = target_components.tid
JOIN component_sequences cs ON target_components.component_id = cs.component_id
LEFT JOIN drug_mechanism dm ON td.tid = dm.tid
WHERE
cs.tax_id = 9606 AND -- 9606 is Homo sapiens
act.standard_units = 'nM' AND
md.max_phase >= 3
GROUP BY md.chembl_id, cs.accession
HAVING
avg_p_activity_value >= 5 AND -- activity (Kd, Ki, EC50, IC50,...) 10 uM or less
cs.component_type = 'PROTEIN'
--LIMIT 1000
;
