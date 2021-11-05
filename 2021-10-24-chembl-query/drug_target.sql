SELECT md.chembl_id AS drug_chembl_id,
md.pref_name AS drug_name,
md.max_phase AS max_phase,
md.indication_class,
td.pref_name AS target_name,
td.organism AS target_organism,
cs.accession AS target_uniprot_ac,
cs.description AS target_description,
cs.organism AS cs_organism,
act.pchembl_value,
act.standard_type AS activity_type,
act.standard_value AS activity_value,
act.standard_units AS activity_units,
assays.assay_type
FROM molecule_dictionary md
JOIN activities act ON md.molregno = act.molregno
JOIN assays ON act.assay_id = assays.assay_id
JOIN target_dictionary td ON assays.tid = td.tid
JOIN target_components ON td.tid = target_components.tid
JOIN component_sequences cs ON target_components.component_id = cs.component_id
WHERE
--act.pchembl_value > 5 AND
cs.tax_id = 9606 AND
act.standard_units = 'nM' AND
md.max_phase = 4
ORDER BY md.chembl_id, cs.accession
LIMIT 1000;
