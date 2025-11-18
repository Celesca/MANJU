'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');

const CalendarIcon = React.forwardRef((props, ref) => /* @__PURE__ */ jsxRuntime.jsx(
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
    children: /* @__PURE__ */ jsxRuntime.jsx(
      "path",
      {
        fillRule: "evenodd",
        stroke: "none",
        d: "M6 2a1 1 0 0 0-1 1v1H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1V3a1 1 0 1 0-2 0v1H7V3a1 1 0 0 0-1-1zm0 5a1 1 0 0 0 0 2h8a1 1 0 1 0 0-2H6z",
        clipRule: "evenodd"
      }
    )
  }
));
CalendarIcon.displayName = "CalendarIcon";

exports.CalendarIcon = CalendarIcon;
//# sourceMappingURL=calendar-icon.cjs.map
