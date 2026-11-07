import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const stage = process.argv[2];
for (const study of ['CO28152', 'CO28154']) {
  const values = JSON.parse(await fs.readFile(`${stage}/${study}.json`, 'utf8'));
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add('Abeta');
  sheet.getRange(`A1:M${values.length}`).values = values;
  sheet.getRange('A1:M2').format.font = {bold: true};
  sheet.getRange('A1:E62').format.columnWidth = 12;
  sheet.getRange('F1:M62').format.columnWidth = 18;
  sheet.getRange('F3:M62').setNumberFormat('0.000000');
  workbook.recalculate();
  const out = await SpreadsheetFile.exportXlsx(workbook);
  await out.save(`${stage}/${study}-biochemistry-cleaned-Abeta.xlsx`);
  console.log(study, 'exported');
}
