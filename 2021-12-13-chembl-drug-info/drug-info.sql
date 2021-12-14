SELECT md.chembl_id AS drug_chembl_id,
md.pref_name AS drug_name,
cs.standard_inchi,
cs.canonical_smiles
FROM molecule_dictionary md
LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE
max_phase >= 3
ORDER BY md.chembl_id
--LIMIT 1000
;
