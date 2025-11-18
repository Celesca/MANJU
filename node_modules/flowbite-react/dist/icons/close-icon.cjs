'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');

const CloseIcon = React.forwardRef((props, ref) => /* @__PURE__ */ jsxRuntime.jsxs(
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
      /* @__PURE__ */ jsxRuntime.jsx(
        "path",
        {
          stroke: "none",
          d: "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"
        }
      )
    ]
  }
));
CloseIcon.displayName = "CloseIcon";

exports.CloseIcon = CloseIcon;
//# sourceMappingURL=close-icon.cjs.map
