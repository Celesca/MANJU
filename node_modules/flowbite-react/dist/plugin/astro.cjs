'use strict';

var index = require('./index.cjs');
var vite = require('./vite.cjs');

var astro = () => ({
  name: index.pluginName,
  hooks: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    "astro:config:setup": async (astro) => {
      var _a;
      (_a = astro.config.vite).plugins || (_a.plugins = []);
      astro.config.vite.plugins.push(vite());
    }
  }
});

module.exports = astro;
//# sourceMappingURL=astro.cjs.map
