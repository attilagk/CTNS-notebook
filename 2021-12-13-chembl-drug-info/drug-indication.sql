SELECT md.chembl_id AS drug_chembl_id,
md.pref_name AS drug_name,
di.mesh_id,
di.mesh_heading,
di.efo_id,
di.efo_term,
di.max_phase_for_ind
FROM molecule_dictionary md
JOIN drug_indication di ON md.molregno = di.molregno
WHERE
max_phase >= 3
ORDER BY md.chembl_id
--LIMIT 1000
;
