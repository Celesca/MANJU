'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');

const MoonIcon = React.forwardRef((props, ref) => /* @__PURE__ */ jsxRuntime.jsx(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "currentColor",
    stroke: "currentColor",
    strokeWidth: 0,
    viewBox: "0 0 20 20",
    ref,
    ...props,
    children: /* @__PURE__ */ jsxRuntime.jsx("path", { stroke: "none", d: "M17.293 13.293A8 8 0 0 1 6.707 2.707a8.001 8.001 0 1 0 10.586 10.586z" })
  }
));
MoonIcon.displayName = "MoonIcon";

exports.MoonIcon = MoonIcon;
//# sourceMappingURL=moon-icon.cjs.map
