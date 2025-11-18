'use strict';

var plugin = require('tailwindcss/plugin.js');
var config = require('./config.cjs');

var index = plugin(
  // plugin
  () => {
  },
  // config
  config.config
);

module.exports = index;
//# sourceMappingURL=index.cjs.map
