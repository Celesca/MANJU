'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');

const OutlineXIcon = React.forwardRef((props, ref) => /* @__PURE__ */ jsxRuntime.jsx(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    viewBox: "0 0 24 24",
    ref,
    ...props,
    children: /* @__PURE__ */ jsxRuntime.jsx("path", { strokeLinecap: "round", strokeLinejoin: "round", d: "M6 18 18 6M6 6l12 12" })
  }
));
OutlineXIcon.displayName = "OutlineXIcon";

exports.OutlineXIcon = OutlineXIcon;
//# sourceMappingURL=outline-x-icon.cjs.map
