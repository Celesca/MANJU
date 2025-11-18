import { jsxs, Fragment, jsx } from 'react/jsx-runtime';
import { StoreInitClient } from './client.js';
import { StoreInitServer } from './server.js';

function StoreInit(props) {
  return /* @__PURE__ */ jsxs(Fragment, { children: [
    /* @__PURE__ */ jsx(StoreInitServer, { ...props }),
    /* @__PURE__ */ jsx(StoreInitClient, { ...props })
  ] });
}
StoreInit.displayName = "StoreInit";

export { StoreInit };
//# sourceMappingURL=index.js.map
