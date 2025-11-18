import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';

const ChevronUpIcon = forwardRef((props, ref) => /* @__PURE__ */ jsx(
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
    children: /* @__PURE__ */ jsx("path", { strokeLinecap: "round", strokeLinejoin: "round", d: "m5 15 7-7 7 7" })
  }
));
ChevronUpIcon.displayName = "ChevronUpIcon";

export { ChevronUpIcon };
//# sourceMappingURL=chevron-up-icon.js.map
