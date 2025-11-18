#!/usr/bin/env node
'use strict';

var main = require('./main.cjs');

const argv = process.argv.slice(2).filter((arg) => arg !== "--");
main.main(argv).catch(console.error);
//# sourceMappingURL=bin.cjs.map
