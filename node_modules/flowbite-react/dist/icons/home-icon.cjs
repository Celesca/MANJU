'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');

const HomeIcon = React.forwardRef((props, ref) => /* @__PURE__ */ jsxRuntime.jsxs(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "currentColor",
    stroke: "currentColor",
    strokeWidth: 0,
    viewBox: "0 0 24 24",
    ref,
    ...props,
    children: [
      /* @__PURE__ */ jsxRuntime.jsx("path", { fill: "none", stroke: "none", d: "M0 0h24v24H0z" }),
      /* @__PURE__ */ jsxRuntime.jsx("path", { stroke: "none", d: "M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" })
    ]
  }
));
HomeIcon.displayName = "HomeIcon";

exports.HomeIcon = HomeIcon;
//# sourceMappingURL=home-icon.cjs.map
