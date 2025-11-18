#!/usr/bin/env node
import { main } from './main.js';

const argv = process.argv.slice(2).filter((arg) => arg !== "--");
main(argv).catch(console.error);
//# sourceMappingURL=bin.js.map
