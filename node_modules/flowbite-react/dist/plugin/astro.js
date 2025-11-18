import { pluginName } from './index.js';
import vitePlugin from './vite.js';

var astro = () => ({
  name: pluginName,
  hooks: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    "astro:config:setup": async (astro) => {
      var _a;
      (_a = astro.config.vite).plugins || (_a.plugins = []);
      astro.config.vite.plugins.push(vitePlugin());
    }
  }
});

export { astro as default };
//# sourceMappingURL=astro.js.map
