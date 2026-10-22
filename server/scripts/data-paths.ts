// Prints the active data root and its children as JSON. Read-only: no app,
// no migration, no folder creation. Run from server/:
//   npx tsx scripts/data-paths.ts
import { locateData } from '../src/dataroot/locate.js';

console.log(JSON.stringify(locateData(), null, 2));
