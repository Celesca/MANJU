import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';

const HomeIcon = forwardRef((props, ref) => /* @__PURE__ */ jsxs(
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
      /* @__PURE__ */ jsx("path", { fill: "none", stroke: "none", d: "M0 0h24v24H0z" }),
      /* @__PURE__ */ jsx("path", { stroke: "none", d: "M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" })
    ]
  }
));
HomeIcon.displayName = "HomeIcon";

export { HomeIcon };
//# sourceMappingURL=home-icon.js.map
