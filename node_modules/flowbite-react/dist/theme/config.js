import { jsx } from 'react/jsx-runtime';
import { StoreInit } from '../store/init/index.js';

function ThemeConfig({ mode }) {
  return /* @__PURE__ */ jsx(StoreInit, { ...{ mode } });
}
ThemeConfig.displayName = "ThemeConfig";

export { ThemeConfig };
//# sourceMappingURL=config.js.map
