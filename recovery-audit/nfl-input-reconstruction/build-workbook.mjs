// Run from a directory with the bundled @oai/artifact-tool dependency available.
import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [input, output, preview] = process.argv.slice(2);
const values = JSON.parse(await fs.readFile(input, 'utf8'));
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('NFL in vivo clean');
sheet.getRange(`A1:I${values.length}`).values = values;
sheet.getRange(`A1:I${values.length}`).format.font = { name: 'Arial', size: 11 };
sheet.getRange('A1:I1').format.font = { bold: true };
sheet.getRange('A1:E61').format.columnWidth = 12;
sheet.getRange('F1:I61').format.columnWidth = 17;
sheet.getRange('F2:I61').setNumberFormat('0.000');
workbook.recalculate();
console.log((await workbook.inspect({kind: 'table', range: "'NFL in vivo clean'!A1:I7", tableMaxRows: 7, tableMaxCols: 9})).ndjson);
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(output);
const image = await workbook.render({sheetName: sheet.name, range: 'A1:I10', scale: 1.5, format: 'png'});
await fs.writeFile(preview, new Uint8Array(await image.arrayBuffer()));
