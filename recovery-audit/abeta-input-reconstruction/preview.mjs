import fs from 'node:fs/promises';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(process.argv[2]));
const image = await wb.render({sheetName:'Abeta',range:'A1:M6',scale:1,format:'png'});
await fs.writeFile(process.argv[3],new Uint8Array(await image.arrayBuffer()));
