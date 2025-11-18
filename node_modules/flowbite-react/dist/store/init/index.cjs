'use strict';

var jsxRuntime = require('react/jsx-runtime');
var client = require('./client.cjs');
var server = require('./server.cjs');

function StoreInit(props) {
  return /* @__PURE__ */ jsxRuntime.jsxs(jsxRuntime.Fragment, { children: [
    /* @__PURE__ */ jsxRuntime.jsx(server.StoreInitServer, { ...props }),
    /* @__PURE__ */ jsxRuntime.jsx(client.StoreInitClient, { ...props })
  ] });
}
StoreInit.displayName = "StoreInit";

exports.StoreInit = StoreInit;
//# sourceMappingURL=index.cjs.map
